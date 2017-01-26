import numpy as np
import scipy
import scipy.linalg
from func import *
from pyscf import gto, dft, scf, ao2mo
from tdfields import *
from cmath import *

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
        self.X = None # AO => LAO
        self.V = None # LAO x current MO
        self.H = None # (ao X ao)  core hamiltonian.
        self.log = []

        # Objects
        self.the_scf  = the_scf_
        self.mol = the_scf_.mol
        self.auxmol_set()
        self.params = dict()
        self.initialcondition()
        self.field = fields(the_scf_, self.params)
        self.field.InitializeExpectation(self.rho,self.C)
        self.prop()
        return

    def auxmol_set(self,auxbas = "weigend"):
        auxmol = gto.Mole()
        auxmol.atom = self.mol.atom
        auxmol.basis = self.mol.basis
        auxmol.build()
        mol = self.mol
        nao = self.n_ao = mol.nao_nr()
        naux = self.n_aux = auxmol.nao_nr()


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

        return

    def get_jk(self, P):

        J = self.get_j(Pt)
        K = self.get_k(Pt)


        return J, K

    def get_j(self,P):
        naux = self.n_aux
        nao = self.n_ao
        rho = np.einsum('ijp,ij->p', self.eri3c, P)
        rho = np.linalg.solve(self.eri2c, rho)
        jmat = np.einsum('p,ijp->ij', rho, self.eri3c)

        return jmat

    def get_k(self,P):
        naux = self.n_aux
        nao = self.n_ao
#        rP = P.real
#        iP = P.imag
        kpj = np.einsum('ijp,jk->ikp', self.eri3c, P)
        pik = np.linalg.solve(self.eri2c, kpj.reshape(-1,naux).T.conj())
        rkmat = np.einsum('pik,kjp->ij', pik.reshape(naux,nao,nao), self.eri3c)

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
        self.params["dt"] = 0.02
        self.params["MaxIter"] = 10000000
        self.params["Model"] = "TDDFT"
        self.params["Method"] = "MMUT"
        self.params["ExDir"] = 1.0
        self.params["EyDir"] = 1.0
        self.params["EzDir"] = 1.0
        self.params["FieldAmplitude"] = 0.001
        self.params["FieldFreq"] = 1.1
        self.params["Tau"] = 0.07
        self.params["tOn"] = 7.0*self.params["Tau"]
        # Here they should be read from disk.
        return

    def InitializeLiouvillian(self):
        '''
        Get an initial Fock matrix.
        '''
        S = self.S = self.the_scf.get_ovlp()
        self.X = MatrixPower(S,-1./2.)
        self.C = self.X # Initial set of orthogonal coordinates.
        self.InitFockBuild()
        self.rhoM12 = self.rho = 0.5*np.diag(self.the_scf.mo_occ)
        return

    def InitFockBuild(self):
        '''
        Using Roothan's equation to build a Fock matrix and initial density matrix
        '''
        n_occ = self.n_occ
        Ne = self.n_e = 2 * n_occ
        err = 100
        it = 0
        self.H = self.the_scf.get_hcore()
        S = self.S
        P = self.the_scf.make_rdm1()
        Veff = dft.rks.get_veff(self.the_scf, None, P)
        self.F = self.H + Veff
        # Roothan's Equation
        # Question: Does Fock matrix needs to be solved in MO basis?
        # for the time being anything goes...
        # MO basis is easier to implement.
        E = self.energy(P)+ self.Enuc
        while (err > 10**-10):
            Fold = self.F
            eigs, C = scipy.linalg.eigh(self.F, self.S)
            #idx = self.eigs.argsort()
            #self.eigs.sort()
            #self.C = self.C[:,idx]

            #self.C = self.C.astype(np.complex)
            #for i in range(self.n_mo):
            #    self.C[:,i] *= np.exp(1.j*np.random.random())
        #    C = self.C * e**(1j * 0.1)
            Pold = P
            P = 2 * np.dot(C[:,:n_occ],C[:,:n_occ].T.conj())
            P = 0.6*Pold + 0.4*P
            P = Ne * P / (TrDot(P, self.S))
            self.F = self.H + dft.rks.get_veff(self.the_scf, None, P)
            Eold = E
            E = self.energy(0.5*P) + self.Enuc
            err = abs(E-Eold)
            if (it%10 ==0):
                print "Iteration:", it,"; Energy:",E,"; Error =",err
            it += 1
        self.eigs = eigs
        self.C = C
        P = 2 * np.dot(self.C[:,:n_occ],self.C[:,:n_occ].T.conj())
        print "Energy:",E
        print "Ne:", TrDot(P, S)
        if (0):
            print "Checking MO=>AO"
            rho = np.diag(0.5*self.the_scf.mo_occ)
            pt = TransMat(rho,self.C,-1)
            print "1--",self.energy(pt)+ self.Enuc
            Ftmp = self.FockBuild(pt)
            print "2--",self.energy(pt)+ self.Enuc
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

    def TDDFTstep(self,time):
        if (self.params["Method"] == "MMUT"):
            # First perform a fock build
            self.FockBuild( TransMat(self.rho, self.C, -1) )
            #print "\nDBG1", self.energy(TransMat(self.rho, self.C, -1))+ self.Enuc
            #print "DBG1c", self.energyc(TransMat(self.rho, self.C, -1))+ self.Enuc
            Fmo = TransMat(self.F,self.C)
            self.eigs, rot = np.linalg.eig(Fmo)
            #print self.eigs
            # rotate the densitites and C into this basis(MO).
            self.rho = TransMat(self.rho, rot)
            self.rhoM12 = TransMat(self.rhoM12, rot)
            self.C = np.dot(self.C , rot)
            #TMPPPP = TransMat(self.rho, self.C, -1)
            #print "\nDBG2", self.energy(TransMat(self.rho, self.C, -1),True)+ self.Enuc
            #print "DBG2c", self.energyc(TransMat(self.rho, self.C, -1),True)+ self.Enuc
            # Make the exponential propagator for the dipole.
            FmoPlusField, IsOn = self.field.ApplyField(Fmo,self.C,time)
            # For exponential propagator:
            w,v = np.linalg.eig(FmoPlusField)
            NewRhoM12 = self.Split_RK4_Step_MMUT(w, v, self.rhoM12, time, self.params["dt"], IsOn)
            NewRho = self.Split_RK4_Step_MMUT(w, v, NewRhoM12, time,self.params["dt"]/2.0, IsOn)
            self.rho = 0.5*(NewRho+(NewRho.T.conj()));
            #print TMPPPP - TransMat(self.rho, self.C, -1)
            #print "\nDBG3", self.energy(TransMat(self.rho, self.C, -1),True) + self.Enuc
            #print "DBG3c", self.energyc(TransMat(self.rho, self.C, -1),True) + self.Enuc
            self.rhoM12 = 0.5*(NewRhoM12+(NewRhoM12.T.conj()))
        elif (self.params["Method"] == "RK4"):
            print "Finish step."
        else:
            raise Exception("Unknown Method...")
        return

    def step(self,time):
        """
        Performs a step
        Updates t, rho, and possibly other things.
        """
        if (self.params["Model"] == "TDDFT"):
            return self.TDDFTstep(time)
        return

    def dipole(self):
        return self.field.Expectation(self.rho, self.C)

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
        print "Complex P energy"
        Pt = 2.0*P
        Ecore = TrDot(Pt, self.H)
        J,K = self.get_jk(Pt)
        Ecoul = 0.5*TrDot(Pt,J)
        nn, Exc, Vx = self.the_scf._numint.nr_rks(self.mol, self.the_scf.grids, self.the_scf.xc, Pt, 0)
        hyb = self.the_scf._numint.hybrid_coeff(self.the_scf.xc, spin=(self.mol.spin>0)+1)
        Exc -= 0.5 * 0.5 * hyb * TrDot(Pt,K)
        if (IfPrint):
                print Ecore, Ecoul, Exc, -0.25*TrDot(Pt,K)
        E = Ecore + Ecoul + Exc
        E = Ecore + Ecoul -0.25*TrDot(Pt,K)
        return E

    def loginstant(self):
        tore = [self.t]+self.dipole().tolist()+[self.energyc(TransMat(self.rho,self.C,-1))+self.Enuc]
        print tore
        return tore

    def prop(self):
        """
        The main tdscf propagation loop.
        """
        iter = 0
        self.t = 0
        while (iter<self.params["MaxIter"]):
            self.step(self.t)
            self.log.append(self.loginstant())
            # Do logging.
            iter = iter + 1
            self.t = self.t + self.params["dt"]
