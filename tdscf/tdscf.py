import numpy as np
import scipy
import scipy.linalg
from func import *
from pyscf import gto, dft, scf, ao2mo
from tdfields import *
from cmath import *
from pyscf import lib
import ctypes

libtdscf = lib.load_library('libtdscf')

class tdscf:
    """
    A general TDSCF object.
    Other types of propagations may inherit from this.

    By default it does
    """
    def __init__(self,the_scf_):
        """
        Args:
            the_scf an SCF object from pyscf (should probably take advantage of complex RKS already in PYSCF)
        Returns:
            Nothing.
        """
        #To be Sorted later
        self.Enuc = the_scf_.e_tot - dft.rks.energy_elec(the_scf_,the_scf_.make_rdm1())[0]
        self.eri3c = None
        self.eri2c = None
        self.n_aux = None
        self.muxo = None
        self.muyo = None
        self.muzo = None
        self.mu0 = None
        #Global numbers
        self.t = 0.0
        self.n_ao = None
        self.n_mo = None
        self.n_occ = None
        self.n_e = None

        #Global Matrices
        self.rho = None # Current MO basis density matrix. (the idempotent (0,1) kind)
        self.rhoM12 = None # For MMUT step
        self.F = None # (AO x AO)
        self.eigs = None # current fock eigenvalues.
        self.S = None # (ao X ao)
        self.C = None # (ao X mo)
        self.Cinv = None #(mo x ao)
        self.X = None # AO => LAO
        self.V = None # LAO x current MO
        self.H = None # (ao X ao)  core hamiltonian.
        self.B = None # for ee2
        self.log = []

        # Objects
        self.the_scf  = the_scf_
        self.mol = the_scf_.mol
        self.auxmol_set()
        self.params = dict()
        self.initialcondition()
        self.field = fields(the_scf_, self.params)
        #self.field.Update(self.Cinv)
        self.field.InitializeExpectation(self.rho,self.C)
        self.CField()
        self.prop()
        return

    def auxmol_set(self,auxbas = "weigend"):
        auxmol = gto.Mole()
        auxmol.atom = self.mol.atom
        auxmol.basis = auxbas
        auxmol.build()
        mol = self.mol
        nao = self.n_ao = mol.nao_nr()
        naux = self.n_aux = auxmol.nao_nr()
        #print nao
        #print naux
        atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
        eri3c = np.empty((nao,nao,naux))
        pi = 0
        for i in range(mol.nbas):
            pj = 0
            for j in range(mol.nbas):
                pk = 0
                for k in range(mol.nbas, mol.nbas+auxmol.nbas):
                    shls = (i, j, k)
                    buf = gto.getints_by_shell('cint3c2e_sph', shls, atm, bas, env)
                    di, dj, dk = buf.shape
                    eri3c[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
                    pk += dk
                pj += dj
            pi += di

        eri2c = np.empty((naux,naux))
        pk = 0
        for k in range(mol.nbas, mol.nbas+auxmol.nbas):
            pl = 0
            for l in range(mol.nbas, mol.nbas+auxmol.nbas):
                shls = (k, l)
                buf = gto.getints_by_shell('cint2c2e_sph', shls, atm, bas, env)
                dk, dl = buf.shape
                eri2c[pk:pk+dk,pl:pl+dl] = buf
                pl += dl
            pk += dk

        self.eri3c = eri3c
        self.eri2c = eri2c

        RSinv = MatrixPower(eri2c,-0.5)
        self.B = np.einsum('ijp,pq->ijq', self.eri3c, RSinv) # (AO,AO,n_aux)

        return

    def get_jk(self, P):
        '''
        Args:
            P: AO density matrix

        Returns:
            J: Coulomb matrix
            K: Exchange matrix
        '''
        J = self.get_j(P)
        K = self.get_k(P)


        return J, K

    def get_j(self,P):
        '''
        Args:
            P: AO density matrix

        Returns:
            J: Coulomb matrix (AO)
        '''
        naux = self.n_aux
        nao = self.n_ao
        rho = np.einsum('ijp,ij->p', self.eri3c, P)
        rho = np.linalg.solve(self.eri2c, rho)
        jmat = np.einsum('p,ijp->ij', rho, self.eri3c)

        return jmat

    def get_k(self,P):
        '''
        Args:
            P: AO density matrix

        Returns:
            K: Exchange matrix (AO)
        '''
        naux = self.n_aux
        nao = self.n_ao
        #rP = np.zeros((nao,nao)).astype(complex)
        #iP = np.zeros((nao,nao)).astype(complex)
        #rP += P.real
        #iP += 1j*P.imag
        kpj = np.einsum('ijp,jk->ikp', self.eri3c, P)
        pik = np.linalg.solve(self.eri2c, kpj.reshape(-1,naux).T.conj())
        rkmat = np.einsum('pik,kjp->ij', pik.reshape(naux,nao,nao), self.eri3c)
        #print 'K(AO)\n',rkmat
        #print 'K(MO)\n',TransMat(rkmat,self.Cinv,-1)
        #kpj = np.einsum('ijp,jk->ikp', self.eri3c, iP)
        #pik = np.linalg.solve(self.eri2c, kpj.reshape(-1,naux).T.conj())
        #ikmat = np.einsum('pik,kjp->ij', pik.reshape(naux,nao,nao), self.eri3c)

        return rkmat #+ ikmat

    def initialcondition(self):
        print '''
        ===================================
        |  Realtime TDSCF module          |
        ===================================
        | J. Parkhill, T. Nguyen          |
        | J. Koh, J. Herr,  K. Yao        |
        ===================================
        | Refs: 10.1021/acs.jctc.5b00262  |
        |       10.1063/1.4916822         |
        ===================================
        '''
        n_ao = self.n_ao = self.the_scf.make_rdm1().shape[0]
        n_mo = self.n_mo = n_ao # should be fixed.
        n_occ = self.n_occ = int(sum(self.the_scf.mo_occ)/2)
        print "n_ao:", n_ao, "n_mo:", n_mo, "n_occ:", n_occ
        self.ReadParams()
        self.InitializeLiouvillian()
        return

    def ReadParams(self):
        '''
        Read the file and fill the params dictionary
        '''
        self.params["dt"] =  0.02
        self.params["MaxIter"] = 5000#1000000000
        self.params["Model"] = "EE2"#"TDDFT"
        self.params["Method"] = "RK4"#"MMUT"
        self.params["ExDir"] = 1.0
        self.params["EyDir"] = 1.0
        self.params["EzDir"] = 1.0
        self.params["FieldAmplitude"] = 0.001
        self.params["FieldFreq"] = 1.1
        self.params["Tau"] = 0.07
        self.params["tOn"] = 7.0*self.params["Tau"]
        self.params["StatusEvery"] = 1000
        # Here they should be read from disk.
        print self.params
        return

    def InitializeLiouvillian(self):
        '''
        Get an initial Fock matrix.
        '''
        S = self.S = self.the_scf.get_ovlp()
        self.X = MatrixPower(S,-1./2.)
        self.H = self.the_scf.get_hcore()
        #self.X = scipy.linalg.fractional_matrix_power(S,-1./2.)
        self.C = self.X.copy() # Initial set of orthogonal coordinates.
        if (self.params["Model"] == "TDDFT"):
            self.InitFockBuild()
        elif(self.params["Model"] == "EE2"):
            self.InitFockBuildC()
        self.rho = 0.5*np.diag(self.the_scf.mo_occ).astype(complex)
        self.rhoM12 = self.rho.copy()

        return

    def InitFockBuildC(self):
        '''
        Transfer H,S,X,B to CPP
        Initialize V matrix and eigs vec in CPP as well
        Make Fockbuild
        '''
        F = np.zeros((self.n_ao,self.n_ao)).astype(complex)
        libtdscf.Initialize(\
        self.H.ctypes.data_as(ctypes.c_void_p),self.S.ctypes.data_as(ctypes.c_void_p),\
        self.X.ctypes.data_as(ctypes.c_void_p),self.B.ctypes.data_as(ctypes.c_void_p),F.ctypes.data_as(ctypes.c_void_p),\
        ctypes.c_int(self.n_ao),ctypes.c_int(self.n_aux),ctypes.c_int(self.n_occ),\
        ctypes.c_double(self.Enuc), ctypes.c_double(self.params["dt"]))

        #print "F\n",F

    def CField(self):

        '''
        setup field components in C
        '''
        mux = 2*self.field.dip_ints[0].astype(complex)
        muy = 2*self.field.dip_ints[1].astype(complex)
        muz = 2*self.field.dip_ints[2].astype(complex)
        nuc = self.field.nuc_dip
        #print "Nuc", nuc

        libtdscf.FieldOn(\
        self.rho.ctypes.data_as(ctypes.c_void_p),\
        nuc.ctypes.data_as(ctypes.c_void_p),\
        mux.ctypes.data_as(ctypes.c_void_p),\
        muy.ctypes.data_as(ctypes.c_void_p),\
        muz.ctypes.data_as(ctypes.c_void_p),\
        ctypes.c_double(self.field.pol[0]), ctypes.c_double(self.field.pol[1]), ctypes.c_double(self.field.pol[2]),\
        ctypes.c_double(self.params["FieldAmplitude"]), ctypes.c_double(self.params["FieldFreq"]),\
        ctypes.c_double(self.params["Tau"]), ctypes.c_double(self.params["tOn"])
        )

        # Bring the field Components
        n = self.n_ao
        muxo = np.zeros((n,n)).astype(complex)
        muyo = np.zeros((n,n)).astype(complex)
        muzo = np.zeros((n,n)).astype(complex)
        mu0 = np.zeros(3)

        libtdscf.InitMu(\
        muxo.ctypes.data_as(ctypes.c_void_p),\
        muyo.ctypes.data_as(ctypes.c_void_p),\
        muzo.ctypes.data_as(ctypes.c_void_p),\
        mu0.ctypes.data_as(ctypes.c_void_p))

        self.muxo = muxo.copy()
        self.muyo = muyo.copy()
        self.muzo = muzo.copy()
        self.mu0 = mu0.copy()

        #print muxo
        #print muyo
        #print muzo
        #print mu0
        #quit()

        return

    def InitFockBuild(self):
        '''
        Using Roothan's equation to build a Fock matrix and initial density matrix
        '''
        n_occ = self.n_occ
        Ne = self.n_e = 2.0 * n_occ
        err = 100
        it = 0
        self.H = self.the_scf.get_hcore()
        S = self.S.copy()
        P = self.the_scf.make_rdm1()
        #Veff = dft.rks.get_veff(self.the_scf, None, P)
        #self.F = self.H + Veff
        J,K = self.get_jk(P)
        self.F = self.H + J - 0.5*K
        # Roothan's Equation
        # Question: Does Fock matrix needs to be solved in MO basis?
        # for the time being anything goes...
        # MO basis is easier to implement.
        E = self.energyc(P)+ self.Enuc
        while (err > 10**-10):
            Fold = self.F
            #eigs, C = scipy.linalg.eig(self.F, self.S)#This routine gives C(LAOxMO); wrong density
            eigs, C = scipy.linalg.eigh(self.F, self.S)#This routine gives C(AOxMO); and right density
            #idx = eigs.argsort()
            #eigs.sort()
            #C = C[:,idx]
            Pold = P
            #P = 2.0 * TransMat(np.dot(C[:,:n_occ],C[:,:n_occ].T.conj()),self.X) #X(LAO,AO) # thus in AO basis
            P = 2.0 * np.dot(C[:,:n_occ],C[:,:n_occ].T.conj()) # AO Basis if scipy.eigh # LAO basis if scipy.eig
            P = 0.4*P + 0.6*Pold
            #print np.dot(self.X,self.X.T) # X is not unitary matrix
            print (TrDot(P, self.S))
            #P = Ne * P / (TrDot(P, self.S)) # AO
            #self.F = self.H + dft.rks.get_veff(self.the_scf, None, P)
            #print TransMat(P,C)
            Eold = E
            J,K = self.get_jk(P)
            self.F = self.H + J - 0.5*K
            E = self.energyc(0.5*P) + self.Enuc # energy uses AO basis
            #print "Rho(MO)\n", TransMat(P,np.dot(self.X.T.conj(),C)) # X.T(AOxLAO)*C(LAOxMO) = (AOxMO)
            #print E
            err = abs(E-Eold)
            if (it%10 ==0):
                print "Iteration:", it,"; Energy:",E,"; Error =",err
            it += 1

        Cinv = np.linalg.inv(C).copy()
        self.eigs = eigs.copy()
        self.C = C.copy()
        self.Cinv = Cinv.copy()

        print "\n\nNe(if right P(AO)):",(TrDot(P, self.S))
        Pm = TransMat(P,Cinv,-1) #C(AOxMO)
        print "Rho(MO)\n",Pm
        print "Ne(MO)", np.trace(Pm)
        Pao = TransMat(Pm,C,-1)
        print "Rho(AO) by TransMat\n",Pao
        print "Rho(AO)\n",P
        print "\n\n"
        # C(AOxMO), X(AO,LAO) -> V(LAOxMO)
        #self.V = np.dot(self.X.T.conj(),self.C)
        P = 2 * np.dot(self.C[:,:n_occ],self.C[:,:n_occ].T.conj()) # C(AOxMO)
        print "Energy:",E
        print "Ne:", TrDot(P, S)
        print "Initial Fock Matrix(AO)\n", self.F
        print "Initial Fock Matrix(MO)\n", TransMat(self.F,Cinv,-1)
        if (0):
            print "Checking MO=>AO"
            rho = np.diag(0.5*self.the_scf.mo_occ)
            pt = TransMat(rho,self.C,-1)
            print "1--",self.energy(pt)+ self.Enuc
            Ftmp = self.FockBuild(pt)
            print "2--",self.energy(pt)+ self.Enuc
        return P


    def InitFockBuilddb(self):
        n_occ = self.n_occ
        Ne = self.n_e = 2.0 * n_occ
        err = 100
        it = 0
        self.H = self.the_scf.get_hcore()
        S = self.S.copy()
        P = self.the_scf.make_rdm1()
        J,K = self.get_jk(P)
        self.F = self.H + J - 0.5*K
        E = self.energyc(P)+ self.Enuc
        while (err > 10**-10):
            Fold = self.F
            eigs, C = scipy.linalg.eigh(self.F,self.S)
            print "C\n",C
            print "P",P
            print "Eigval",eigs
            #idx = eigs.argsort()
            #eigs.sort()
            #C = C[:,idx]
            Pold = P
            P = 2.0 * np.dot(C[:,:n_occ],C[:,:n_occ].T.conj()) # AO Basis
            P = 0.4*P + 0.6*Pold
            P = Ne * P / TrDot(P,S)

            Eold = E
            J,K = self.get_jk(P)
            self.F = self.H + J - 0.5*K
            E = self.energyc(0.5*P) + self.Enuc
            err = abs(E-Eold)
            if (it%10 ==0):
                print "Iteration:", it,"; Energy:",E,"; Error =",err
            it += 1
        Cinv = np.linalg.inv(C).copy()
        self.eigs = eigs.copy()
        self.C = C.copy()
        self.Cinv = Cinv.copy()

        print "\n\nNe(if right P(AO)):",(TrDot(P, self.S))
        Pm = TransMat(P,Cinv,-1) #C(AOxMO)
        print "Rho(MO)\n",Pm
        print "Ne(MO)", np.trace(Pm)
        Pao = TransMat(Pm,C,-1)
        print "Rho(AO) by TransMat\n",Pao
        print "Rho(AO)\n",P
        print "\n\n"
        # C(AOxMO), X(AO,LAO) -> V(LAOxMO)
        #self.V = np.dot(self.X.T.conj(),self.C)
        P = 2 * np.dot(self.C[:,:n_occ],self.C[:,:n_occ].T.conj()) # C(AOxMO)
        print "Energy:",E
        print "Ne:", TrDot(P, S)
        print "Initial Fock Matrix(AO)\n", self.F
        print "Initial Fock Matrix(MO)\n", TransMat(self.F,Cinv,-1)

        quit()
        return P

    def FockBuild(self,P):
        """
        Updates self.F given current self.rho (both complex.)
        Args:
            P = AO density matrix.
        Returns:
            Fock matrix(ao) . Updates self.F
        """
        self.F = self.H + dft.rks.get_veff(self.the_scf, None, 2.0*P)
        return  self.F

    def FockBuildc(self,P):
        """
        Updates self.F given current self.rho (both complex.)
        Args:
            P = AO density matrix.
        Returns:
            Fock matrix(ao) . Updates self.F
        """
        Pt = 2.0*P
        J,K = self.get_jk(Pt)
        self.F = self.H + 0.5*(J+J.T.conj()) - 0.5*(0.5*(K + K.T.conj()))
        return  self.F

    def Fockbuildee2(self,P):
        '''
        Updates Fock matrix using ee2 Fockbuild routine
        V_ is empty, F is empty
        need V
        Updating F, V_, V, C, eigs
        make Vprime
        '''
        # Rotate all matrix
        RhoT = self.Rho.T.copy() # Complex
        V_ = np.zeros((n,n)).astype(complex)
        VT = self.V.T.copy() # Complex
        CT = self.C.T.copy() # Complex
        XT = self.X.T.copy()
        HT = self.H.T.copy()
        FT = np.zeros((n,n)).astype(complex)


        libtdscf.setupFock(RhoT.ctypes.data_as(ctypes.c_void_p),V_.ctypes.data_as(ctypes.c_void_p),VT.ctypes.data_as(ctypes.c_void_p))
        V_T = V_.T.copy()
        #cx_mat& Rho, cx_mat& V_, cx_mat& V, cx_mat& C, mat& X, mat& H, cube& BpqR, vec& eigs, int n2, int n_aux

        return F

    def Split_RK4_Step_MMUT(self, w, v , oldrho , time, dt ,IsOn):
        Ud = np.exp(w*(-0.5j)*dt);
        U = TransMat(np.diag(Ud),v,-1)
        RhoHalfStepped = TransMat(oldrho,U,-1)
        # If any TCL propagation occurs...

        DontDo="""
        SplitLiouvillian( RhoHalfStepped, k1,tnow,IsOn);
        v2 = (dt/2.0) * k1;
        v2 += RhoHalfStepped;
        SplitLiouvillian(  v2, k2,tnow+(dt/2.0),IsOn);
        v3 = (dt/2.0) * k2;
        v3 += RhoHalfStepped;
        SplitLiouvillian(  v3, k3,tnow+(dt/2.0),IsOn);
        v4 = (dt) * k3;
        v4 += RhoHalfStepped;
        SplitLiouvillian(  v4, k4,tnow+dt,IsOn);
        newrho = RhoHalfStepped;
        newrho += dt*(1.0/6.0)*k1;
        newrho += dt*(2.0/6.0)*k2;
        newrho += dt*(2.0/6.0)*k3;
        newrho += dt*(1.0/6.0)*k4;
        newrho = U*newrho*U.t();
        """
        newrho = TransMat(RhoHalfStepped,U,-1)
#        print U.dot(U.T.conj())
#        print newrho
        return newrho

    def rhodot(self,rho,time,F):
        '''
        Args:
            rho: MO
            time: au
            F: MO
        Returns:
            kt: rhodot(MO)
        '''
        Ft, IsOn = self.field.ApplyField(F,self.Cinv,time)
        #print Ft
        kt = -1.j*(np.dot(Ft,rho) - np.dot(rho,Ft))
        #print kt
        return kt

    def TDDFTstep(self,time):
        #self.rho (MO basis)
        if (self.params["Method"] == "MMUT"):
            #Smo = TransMat(self.S,self.Cinv,-1)
            # First perform a fock build
            self.FockBuildc( TransMat(self.rho, self.C, -1) )
            #Fmo = TransMat(self.F,self.Cinv,-1) # AO -> MO requires Cinv
            #self.eigs, rot = np.linalg.eig(self.F)
            #self.rho = TransMat(self.rho , rot)
            #self.rhoM12 = TransMat(self.rhoM12 , rot)


            self.eigs, rot = scipy.linalg.eigh(self.F,self.S)
            Fmo = np.diag(self.eigs).astype(complex)
            self.eigs, rot = np.linalg.eig(Fmo)
            #print self.eigs
            #print rot
            #print self.rho
            #print "C",self.C
            self.C = np.dot(self.C , rot)
            #print "C",self.C
            self.Cinv = np.linalg.inv(self.C)
            #print self.rho
            self.rho = TransMat(self.rho , rot)
            self.rhoM12 = TransMat(self.rhoM12 , rot)
            #print self.rho
            #Fmo = np.diag(self.eigs).astype(complex)
            FmoPlusField, IsOn = self.field.ApplyField(Fmo,self.C,time)
            # rotate the densitites and C into this basis from MO.
            # rot(MOxTO) -> Crot(AOxTO)
            w,v = scipy.linalg.eig(FmoPlusField)
            #print "eigval\n",w
            #print "eigvec\n",v
            # Split RK4 Step MMUT
            #self.rho = TransMat(self.rho , rot)
            #self.rhoM12 = TransMat(self.rhoM12 , rot)
            # Make the exponential propagator for the dipole.
            # C (AOxMO)
            #FmoPlusField, IsOn = self.field.ApplyField(Fmo,self.C,time)
            # For exponential propagator:
            #w,v = np.linalg.eig(FmoPlusField)#,self.S)
            #w,v = scipy.linalg.eig(FmoPlusField,self.S)
            #print FmoPlusField
            NewRhoM12 = self.Split_RK4_Step_MMUT(w, v, self.rhoM12, time, self.params["dt"], IsOn)
            NewRho = self.Split_RK4_Step_MMUT(w, v, NewRhoM12, time,self.params["dt"]/2.0, IsOn)
            self.rho = 0.5*(NewRho+(NewRho.T.conj()));
            self.rhoM12 = 0.5*(NewRhoM12+(NewRhoM12.T.conj()))
            #print "RHO(MO)\n",self.rho
            #quit()
        elif (self.params["Method"] == "RK4"):
            dt = self.params["dt"]
            rho = self.rho.copy()
            C = self.C.copy()
            # First perform a fock build
            # MO -> AO requires C
            self.FockBuildc( TransMat(rho,C, -1) )
            #Fmo = TransMat(self.F,self.Cinv,-1) # AO -> MO requires Cinv NOT A NORM CONSERVING TRANSFORMATION, HENCE WRONG EIGVALUES.
            Fmo = TransMat(self.F,C)# C will F(AO) to F(MO)
            k1 = self.rhodot(rho,time,Fmo)
            v2 = (0.5 * dt)*k1
            v2 += rho
            #print Fmo
            self.FockBuildc( TransMat(v2, C, -1) )
            Fmo = TransMat(self.F,C)
            k2 = self.rhodot(v2, time + 0.5*dt,Fmo)
            v3 = (0.5 * dt)*k2
            v3 += rho

            self.FockBuildc( TransMat(v3, C, -1) )
            Fmo = TransMat(self.F,C)
            k3 = self.rhodot(v3, time + 0.5*dt,Fmo)
            v4 = (1.0 * dt)*k3
            v4 += rho

            self.FockBuildc( TransMat(v4, C, -1) )
            Fmo = TransMat(self.F,C)
            k4 = self.rhodot(v4, time + dt,Fmo)

            drho = dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
            self.rho += drho
            self.rho = 0.5 * (self.rho + self.rho.T.conj())
            #print self.F
            #print Fmo
            #print "t",time, "\n",self.rho
            #print np.trace(self.rho)
        else:
            raise Exception("Unknown Method...")
        return

    def EE2step(self,time):

        if (self.params["Method"] == "MMUT"):
            libtdscf.MMUT_step(self.rho.ctypes.data_as(ctypes.c_void_p), self.rhoM12.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(time))
            # Rho and RhoM12 is updated

        elif (self.params["Method"] == "RK4"):
            #
            newrho = self.rho.copy()
            libtdscf.TDTDAstep(newrho.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(time))
            self.rho = newrho.copy()
            #print self.rho

        #quit()



        return

    def step(self,time):
        """
        Performs a step
        Updates t, rho, and possibly other things.
        """
        if (self.params["Model"] == "TDDFT"):
            return self.TDDFTstep(time)
        elif(self.params["Model"] == "EE2"):
            return self.EE2step(time)
        return




    def dipole(self):
        # self.rho (MO), self.C (AOxMO)
        dipole = [TrDot(self.rho,self.muxo),TrDot(self.rho,self.muyo),TrDot(self.rho,self.muzo)] - self.mu0
        #print dipole.real

        return dipole.real
        #return self.field.Expectation(self.rho, self.C)

    def energy(self,P,IfPrint=False):
        """
        This routine should not re-calculate the Fock matrix (like yesterday... )
        """
        print "real P energy"
        Pt = 2.0*P
        Ecore = TrDot(Pt, self.H)
        J,K = scf.hf.get_jk(self.mol,Pt)
        Ecoul = 0.5*TrDot(Pt,J)
        nn, Exc, Vx = self.the_scf._numint.nr_rks(self.mol, self.the_scf.grids, self.the_scf.xc, Pt, 0)
        hyb = self.the_scf._numint.hybrid_coeff(self.the_scf.xc, spin=(self.mol.spin>0)+1)
        Exc -= 0.5 * 0.5 * hyb * TrDot(Pt,K)
        if (IfPrint):
                print Ecore, Ecoul, Exc
        E = Ecore + Ecoul + Exc
        return E

    def energyc(self,P,IfPrint=False):
        """
        This routine should not re-calculate the Fock matrix (like yesterday... )
        """
        #print "Complex P energy"
        Pt = 2.0*P
        Ecore = TrDot(Pt, self.H)
        J,K = self.get_jk(Pt)
        Ecoul = 0.5*TrDot(Pt,J)
        #nn, Exc, Vx = self.the_scf._numint.nr_rks(self.mol, self.the_scf.grids, self.the_scf.xc, Pt, 0)
        #hyb = self.the_scf._numint.hybrid_coeff(self.the_scf.xc, spin=(self.mol.spin>0)+1)
        #Exc -= 0.5 * 0.5 * hyb * TrDot(Pt,K)
        if (IfPrint):
                print Ecore, Ecoul, -0.25*TrDot(Pt,K)
        #E = Ecore + Ecoul + Exc #RKS Energy
        E = Ecore + Ecoul -0.25*TrDot(Pt,K)# HF Energy
        return E.real

    def loginstant(self,iter):
        np.set_printoptions(precision = 7)
        tore = str(self.t)+" "+str(self.dipole().real).rstrip(']').lstrip('[]')+ " " +str(self.energyc(TransMat(self.rho,self.C,-1),False)+self.Enuc)
        #print tore
        if iter%self.params["StatusEvery"] ==0:
            print 't:', self.t, "    Energy:",self.energyc(TransMat(self.rho,self.C,-1)).real+self.Enuc
            print('Dipole moment(X, Y, Z, au): %8.5f, %8.5f, %8.5f' %(self.dipole().real[0],self.dipole().real[1],self.dipole().real[2]))
        return tore

    def prop(self):
        """
        The main tdscf propagation loop.
        """

        iter = 0
        self.t = 0
        f = open('log.dat','a')
        #w,v = scipy.linalg.eigh(self.H,self.S)
        #print "H eigval",w
        #w,v = scipy.linalg.eigh(self.F,self.S)
        #w,v = np.linalg.eigh(self.F)#,self.S)
        #print "EigenValues:", w
        #print "Energy Gap (eV)",abs(w[0]-w[1])*27.2114
        print "\n\nPropagation Begins"
        while (iter<self.params["MaxIter"]):
            self.step(self.t)
            #self.log.append(self.loginstant(iter))
            f.write(self.loginstant(iter)+"\n")
            # Do logging.
            iter = iter + 1
            self.t = self.t + self.params["dt"]

        f.close()
