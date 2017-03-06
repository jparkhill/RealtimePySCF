import numpy as np
import scipy
import scipy.linalg
from func import *
from pyscf import gto, dft, scf, ao2mo
from tdfields import *
from cmath import *
from pyscf import lib
import ctypes

#libtdscf = lib.load_library('libtdscf')

FsPerAu = 0.0241888

class tdscf:
    """
    A general TDSCF object.
    Other types of propagations may inherit from this.

    By default it does
    """
    def __init__(self,the_scf_,prm=None,output = 'log.dat', prop_=True):
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
        self.hyb = the_scf_._numint.hybrid_coeff(the_scf_.xc, spin=(the_scf_.mol.spin>0)+1)
        #Global numbers
        self.t = 0.0
        self.n_ao = None
        self.n_mo = None
        self.n_occ = None
        self.n_virt = None
        self.n_e = None
        #Global Matrices
        self.rho = None # Current MO basis density matrix. (the idempotent (0,1) kind)
        self.rhoM12 = None # For MMUT step
        self.F = None # (LAO x LAO)
        self.eigs = None # current fock eigenvalues.
        self.S = None # (ao X ao)
        self.C = None # (ao X mo)
        self.X = None # AO, LAO
        self.V = None # LAO, current MO
        self.H = None # (ao X ao)  core hamiltonian.
        self.B = None # for ee2
        self.log = []
        # Objects
        self.the_scf  = the_scf_
        self.mol = the_scf_.mol
        # Should only do this if you need the integrals.
        self.params = dict()
        self.auxmol_set()
        self.initialcondition(prm)
        self.field = fields(the_scf_, self.params)
        self.field.InitializeExpectation(self.rho, self.C)
        self.rho0 = self.rho.copy()
        np.set_printoptions(precision = 7)
        if (prop_):
            self.prop(output)
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

    def FockBuildc(self,P):
        """
        Updates self.F given current self.rho (both complex.)
        Fock matrix with HF
        Args:
            P = LAO density matrix.
        Returns:
            Fock matrix(lao) . Updates self.F
        """
        Pt = 2.0*TransMat(P,self.X,-1)
        J,K = self.get_jk(Pt)
        return  TransMat(self.H + 0.5*(J+J.T.conj()) - 0.5*(0.5*(K + K.T.conj())),self.X)

    def FockDFTbuild(self,P):
        '''
        Updates self.F given current self.rho (both complex.)
        Fock matrix with DFT
        Args:
            P = LAO density matrix.
        Returns:
            Fock matrix(lao) . Updates self.F
        '''
        Pt = 2.0*TransMat(P,self.X,-1) # to AO
        J = self.get_j(Pt)
        Vxc = self.get_vxc(Pt) # Include the Hybrid with K matrix
        Veff = J + Vxc
        self.F = TransMat(self.H + 0.5*(Veff + Veff.T.conj()),self.X)
        return self.F

    def get_vxc(self,P):
        '''
        Args:
            P: AO density matrix

        Returns:
            Vxc: Exchange and Correlation matrix (AO)
        '''
        K = self.get_k(P)
        nn, Exc, Vxc = self.the_scf._numint.nr_rks(self.mol, self.the_scf.grids, self.the_scf.xc, P.real, 0)
        Vxc = Vxc.astype(complex)
        Vxc += -0.5 * self.hyb * K
        return Vxc

    def get_jk(self, P):
        '''
        Args:
            P: AO density matrix
        Returns:
            J: Coulomb matrix
            K: Exchange matrix
        '''
        return self.get_j(P), self.get_k(P)

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
        kpj = np.einsum('ijp,jk->ikp', self.eri3c, P)
        pik = np.linalg.solve(self.eri2c, kpj.reshape(-1,naux).T.conj())
        rkmat = np.einsum('pik,kjp->ij', pik.reshape(naux,nao,nao), self.eri3c)
        return rkmat

    def initialcondition(self,prm):
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
        self.n_virt = self.n_mo - self.n_occ
        print "n_ao:", n_ao, "n_mo:", n_mo, "n_occ:", n_occ
        self.ReadParams(prm)
        self.InitializeLiouvillian()
        return

    def ReadParams(self,prm):
        '''
        Set Defaults, Read the file and fill the params dictionary
        '''
        self.params["Model"] = "TDDFT" #"TDHF"; the difference of Fock matrix and energy
        self.params["Method"] = "MMUT"#"MMUT"
        self.params["BBGKY"]=0
        self.params["TDCIS"]=1

        self.params["dt"] =  0.02
        self.params["MaxIter"] = 15000

        self.params["ExDir"] = 1.0
        self.params["EyDir"] = 1.0
        self.params["EzDir"] = 1.0
        self.params["FieldAmplitude"] = 0.01
        self.params["FieldFreq"] = 0.9202
        self.params["Tau"] = 0.07
        self.params["tOn"] = 7.0*self.params["Tau"]
        self.params["ApplyImpulse"] = 1
        self.params["ApplyCw"] = 0

        self.params["StatusEvery"] = 5000
        self.params["Print"]=0
        # Here they should be read from disk.
        if(prm != None):
            for line in prm.splitlines():
                s = line.split()
                if len(s) > 1:
                    if s[0] == "MaxIter" or s[0] == str("ApplyImpulse") or s[0] == str("ApplyCw") or s[0] == str("StatusEvery"):
                        self.params[s[0]] = int(s[1])
                    elif s[0] == "Model" or s[0] == "Method":
                        self.params[s[0]] = s[1]
                    else:
                        self.params[s[0]] = float(s[1])

        print "============================="
        print "         Parameters"
        print "============================="
        print "Model:", self.params["Model"]
        print "Method:", self.params["Method"]
        print "dt:", self.params["dt"]
        print "MaxIter:", self.params["MaxIter"]
        print "ExDir:", self.params["ExDir"]
        print "EyDir:", self.params["EyDir"]
        print "EzDir:", self.params["EzDir"]
        print "FieldAmplitude:", self.params["FieldAmplitude"]
        print "FieldFreq:", self.params["FieldFreq"]
        print "Tau:", self.params["Tau"]
        print "tOn:", self.params["tOn"]
        print "ApplyImpulse:", self.params["ApplyImpulse"]
        print "ApplyCw:", self.params["ApplyCw"]
        print "=============================\n\n"

        return

    def InitializeLiouvillian(self):
        '''
        Get an initial Fock matrix.
        '''
        S = self.S = self.the_scf.get_ovlp()
        self.X = MatrixPower(S,-1./2.)
        self.V = np.eye(self.n_ao)
        #self.X = scipy.linalg.fractional_matrix_power(S,-1./2.)
        self.C = self.X.copy() # Initial set of orthogonal coordinates.
        self.InitFockBuild() # updates self.C
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

    def Field(self):
        mux = 2*self.field.dip_ints[0].astype(complex)
        muy = 2*self.field.dip_ints[1].astype(complex)
        muz = 2*self.field.dip_ints[2].astype(complex)
        nuc = self.field.nuc_dip
        self.muxo = muxo.copy()
        self.muyo = muyo.copy()
        self.muzo = muzo.copy()
        self.mu0 = mu0.copy()
        return

    def CField(self):
        '''
        setup field components in C
        '''
        mux = 2*self.field.dip_ints[0].astype(complex)
        muy = 2*self.field.dip_ints[1].astype(complex)
        muz = 2*self.field.dip_ints[2].astype(complex)
        nuc = self.field.nuc_dip
        libtdscf.FieldOn(\
        self.rho.ctypes.data_as(ctypes.c_void_p),\
        nuc.ctypes.data_as(ctypes.c_void_p),\
        mux.ctypes.data_as(ctypes.c_void_p),\
        muy.ctypes.data_as(ctypes.c_void_p),\
        muz.ctypes.data_as(ctypes.c_void_p),\
        ctypes.c_double(self.field.pol[0]), ctypes.c_double(self.field.pol[1]), ctypes.c_double(self.field.pol[2]),\
        ctypes.c_double(self.params["FieldAmplitude"]), ctypes.c_double(self.params["FieldFreq"]),\
        ctypes.c_double(self.params["Tau"]), ctypes.c_double(self.params["tOn"]),\
        ctypes.c_int(self.params["ApplyImpulse"]), ctypes.c_int(self.params["ApplyCw"])
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
        return

    def InitFockBuild(self):
        '''
        Using Roothan's equation to build a Fock matrix and initial density matrix
        Returns:
            self consistent density in Lowdin basis.
        '''
        n_occ = self.n_occ
        Ne = self.n_e = 2.0 * n_occ
        err = 100
        it = 0
        self.H = self.the_scf.get_hcore()
        S = self.S.copy()
        P = TransMat(self.the_scf.make_rdm1(), self.X)
        Plao = TransMat(P,self.X,1)
        self.F = self.FockBuildc(Plao)
        Plao_old = TransMat(P,self.X,1)
        E = self.energy(Plao)+ self.Enuc
        while (err > 10**-10):
            # Diagonalize F in the lowdin basis
            self.eigs, self.V = np.linalg.eig(self.F)
            idx = self.eigs.argsort()
            self.eigs.sort()
            self.V = self.V[:,idx].copy()
            # Fill up the density in the MO basis and then Transform back
            Pmo = 0.5*np.diag(self.the_scf.mo_occ).astype(complex)
            Plao = TransMat(Pmo,self.V,-1)
            #print "Ne", np.trace(Plao), np.trace(Pmo)
            Eold = E
            if (self.params["Model"] == "TDHF" or self.params["Model"] == "BBGKY"):
                self.F = self.FockBuildc(Plao)
            elif self.params["Model"] == "TDDFT":
                self.FockDFTbuild(Plao)
            else:
                raise Exception("NoSCF")
            E = self.energy(Plao)
            err = abs(E-Eold)
            print "Iteration:", it,"; Energy:",E,"; Error =",err
            it += 1
        Pmo = 0.5*np.diag(self.the_scf.mo_occ).astype(complex)
        Plao = TransMat(Pmo,self.V,-1)
        self.C = np.dot(self.X,self.V)
        self.rho = TransMat(Plao,self.V)
        self.rhoM12 = TransMat(Plao,self.V)
        print "Ne", np.trace(Plao), np.trace(self.rho),
        print "Energy:",E
        print "Initial Fock Matrix(MO)\n", TransMat(self.F,self.V)
        return Plao

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
            rho: LAO basis density
            time: au
            F: Orig. MO basis fock.
        Returns:
            kt: rhodot(MO)
        '''
        raise Exception("Not fixed")
        print "F",F
        print "Rho", rho
        Ft, IsOn = self.field.ApplyField(F,self.C, time)
        tmp = -1.j*(np.dot(Ft,rho) - np.dot(rho,Ft))
        print "Ft",Ft
        print "tmp[0,1]", tmp[0,1], tmp[1,0], tmp[0,1]/rho[0,1]
        return -1.j*(np.dot(Ft,rho) - np.dot(rho,Ft))

    def TDDFTstep(self,time):
        #self.rho (MO basis)
        if (self.params["Method"] == "MMUT"):
            self.F = self.FockBuildc(TransMat(self.rho,self.V,-1)) # is LAO basis
            self.F = np.conj(self.F)
            Fmo_prev = TransMat(self.F, self.V)
            self.eigs, rot = np.linalg.eig(Fmo_prev)
            # print Fmo_prev, rot
            # Rotate all the densities into the current fock eigenbasis.
            #roti = np.linalg.inv(rot)
            self.rho = TransMat(self.rho, rot)
            self.rhoM12 = TransMat(self.rhoM12, rot)
            self.V = np.dot(self.V , rot)
            self.C = np.dot(self.X , self.V)
            # propagation is done in the current eigenbasis.
            Fmo = np.diag(self.eigs).astype(complex)
            #Fmo = TransMat(self.F,self.V)
            # Check that the energy is still correct.
            Hmo = TransMat(self.H,self.C)
            FmoPlusField, IsOn = self.field.ApplyField(Fmo,self.C,time)
            #print "Fmo + FIeld\n",FmoPlusField
            w,v = scipy.linalg.eig(FmoPlusField)
            NewRhoM12 = self.Split_RK4_Step_MMUT(w, v, self.rhoM12, time, self.params["dt"], IsOn)
            NewRho = self.Split_RK4_Step_MMUT(w, v, NewRhoM12, time,self.params["dt"]/2.0, IsOn)
            self.rho = 0.5*(NewRho+(NewRho.T.conj()));
            self.rhoM12 = 0.5*(NewRhoM12+(NewRhoM12.T.conj()))
        elif (self.params["Method"] == "RK4"):
            raise Exception("Broken.")
            dt = self.params["dt"]
            rho = self.rho.copy()
            C = self.C.copy()
            # First perform a fock build
            # MO -> AO requires C
            self.FockBuildc( TransMat(rho,C, -1) )
            #Fmo = TransMat(self.F,self.Cinv,-1) # AO -> MO requires Cinv NOT A NORM CONSERVING TRANSFORMATION, HENCE WRONG EIGVALUES.
            Fmo = TransMat(self.F,C)# C will F(AO) to F(MO)
            if (1):
                eigs, rot = np.linalg.eig(Fmo)
                print "Eigs", eigs

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
        if (self.params["Model"] == "TDDFT" or self.params["Model"] == "TDHF"):
            return self.TDDFTstep(time)
        elif(self.params["Model"] == "EE2"):
            return self.EE2step(time)
        return

    def dipole(self):
        # self.rho (MO), self.C (AOxMO)
        #dipole = [TrDot(self.rho,self.muxo),TrDot(self.rho,self.muyo),TrDot(self.rho,self.muzo)] - self.mu0
        #quit()

        #print dipole.real

        #return dipole.real
        return self.field.Expectation(self.rho, self.C)

    def energy(self,Plao,IfPrint=False):
        """
        P: Density in LAO basis.
        """
        if (self.params["Model"] == "TDHF" or self.params["Model"] == "BBGKY"):
            Hlao = TransMat(self.H,self.X)
            return (self.Enuc+np.trace(np.dot(Plao,Hlao+self.F))).real
        elif self.params["Model"] == "TDDFT":
            Hlao = TransMat(self.H,self.X)
            P = TransMat(Plao,self.X,-1)
            J = self.get_j(2*P)
            nn, Exc, Vx = self.the_scf._numint.nr_rks(self.mol, self.the_scf.grids, self.the_scf.xc, 2*P.real, 0)
            Exc -= 0.5 * 0.5 * self.hyb * TrDot(P,self.get_k(2*P))
            EH = TrDot(Plao,2*Hlao)
            EJ = TrDot(P,J)
            E = EH + EJ + Exc + self.Enuc
            return E.real
        else :
            return 0.0

    def loginstant(self,iter):
        """
        time is logged in atomic units.
        """
        tore = str(self.t)+" "+str(self.dipole().real).rstrip(']').lstrip('[]')+ " " +str(self.energy(TransMat(self.rho,self.V,-1),False))+" "+str(np.trace(self.rho))
        #tore = str(self.t)+" "+str(np.sin(0.5*np.ones(3)*self.t)).rstrip(']').lstrip('[]')+ " " +str(self.energyc(TransMat(self.rho,self.C,-1),False)+self.Enuc)
        #print tore
        if iter%self.params["StatusEvery"] ==0:
            print 't:', self.t*FsPerAu, " (Fs)  Energy:",self.energy(TransMat(self.rho,self.V,-1)), " Tr ",(np.trace(self.rho))
            print('Dipole moment(X, Y, Z, au): %8.5f, %8.5f, %8.5f' %(self.dipole().real[0],self.dipole().real[1],self.dipole().real[2]))
        return tore

    def prop(self,output):
        """
        The main tdscf propagation loop.
        """
        iter = 0
        self.t = 0
        f = open(output,'a')
        print "Energy Gap (eV)",abs(self.eigs[self.n_occ]-self.eigs[self.n_occ-1])*27.2114
        print "\n\nPropagation Begins"
        while (iter<self.params["MaxIter"]):
            self.step(self.t)
            #self.log.append(self.loginstant(iter))
            f.write(self.loginstant(iter)+"\n")
            # Do logging.
            iter = iter + 1
            self.t = self.t + self.params["dt"]
        f.close()
