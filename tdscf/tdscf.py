import numpy as np
import scipy
import scipy.linalg
from func import *
from pyscf import gto, dft, scf, ao2mo
#from tdfields import *
from cmath import *

class tdscf:
    """
    A general TDSCF object.
    Other types of propagations may inherit from this.

    By default it does
    """
    def __init__(self,the_scf_,mol):
        """
        Args:
            the_scf an SCF object from pyscf (should probably take advantage of complex RKS already in PYSCF)
        Returns:
            Nothing.
        """
        #To be Sorted later
        self.Enuc = the_scf_.e_tot - dft.rks.energy_elec(the_scf_,the_scf_.make_rdm1())[0]
        #Global numbers
        self.t = 0.0
        self.n_ao = None
        self.n_mo = None
        self.n_occ = None

        #Global Matrices
        self.rho = None # Current MO basis density matrix.
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
        self.mol = mol
        self.params = dict()
        self.initialcondition()
        #self.field = fields(the_scf_, self.params)
        self.prop()
        return

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
        return

    def InitializeLiouvillian(self):
        '''
        Get an initial Fock matrix.
        '''
        S = self.S = self.the_scf.get_ovlp()
        self.X = MatrixPower(S,-1./2.)
        rhoAO = self.InitFockBuild()
        return

    def InitFockBuild(self):
        '''
        Using Roothan's equation to build a Fock matrix and initial density matrix
        '''
        n_occ = self.n_occ
        Ne = 2 * n_occ
        err = 100
        it = 0
        # Making rotation matrix X(|AO><MO|)
        self.H = self.the_scf.get_hcore()
        S = self.the_scf.get_ovlp()
        P = self.the_scf.make_rdm1()
        Veff = dft.rks.get_veff(self.the_scf, None, P)
        self.F = self.H + Veff
        E = self.energy(P)+ self.Enuc
        # Roothan's Equation
        # Question: Does Fock matrix needs to be solved in MO basis?
        # for the time being anything goes...
        # MO basis is easier to implement.
        while (err > 10**-10):
            Fold = self.F
            self.eigs, self.C = scipy.linalg.eigh(self.F, S)
            #self.C = self.C.astype(np.complex)
            #for i in range(self.n_mo):
            #    self.C[:,i] *= np.exp(1.j*np.random.random())
        #    C = self.C * e**(1j * 0.1)
            # P/2 = C*C.t() for Occ orbitals
            Pold = P
            #P = 2 * np.dot(self.C[:,:n_occ],np.transpose(self.C[:,:n_occ]))
            P = 2 * np.dot(self.C[:,:n_occ],(self.C[:,:n_occ].T))
            P = 0.6*Pold + 0.4*P
            P = Ne * P / (TrDot(P, S))
            self.F = self.H + dft.rks.get_veff(self.the_scf, None, P)
            Eold = E
            E = self.energy(P) + self.Enuc
            err = abs(E-Eold)#abs(sum(sum(self.F-Fold)))
            if (it%10 ==0):
                print "Iteration:", it,"; Energy:",E,"; Error =",err
            it += 1
        print "Energy:",E
        print "Ne:", TrDot(P, S)
        return P

    def FockBuild(self):
        """
        Updates self.F given current self.rho (both complex.)
        """
        return  self.H

    def TDDDFTstep(self):
        if (self.params["Method"] == "MMUT"):
            # First perform a fock build, and rotate the MO densities into the current basis.
            print ' '
        elif (self.params["Method"] == "RK4"):
            print "Finish step."
        else:
            raise Exception("Unknown Method...")
        return

    def step(self):
        """
        Performs a step
        Updates t, rho, and possibly other things.
        """
        if (self.params["Model"] == "TDDFT"):
            return self.TDDFTstep()
        return

    def dipole(self):
        return self.fields.Expectation(self.rho)

    def energy(self,P):

        Ecore = TrDot(P, self.H)

        J,K = scf.hf.get_jk(self.mol,P)
        Ecoul = 0.5*TrDot(P,J)

        nn, Exc, Vx = self.the_scf._numint.nr_rks(self.mol, self.the_scf.grids, self.the_scf.xc, P, 0)
        hyb = self.the_scf._numint.hybrid_coeff(self.the_scf.xc, spin=(self.mol.spin>0)+1)
        Exc -= 0.5 * 0.5 * hyb * TrDot(P,K)

        E = Ecore + Ecoul + Exc
        return E


    def loginstant(self):
        tore = [self.t]+self.dipole()+[self.energy()]
        return

    def prop(self):
        """
        The main tdscf propagation loop.
        """
        iter = 0
        while (iter<self.params["MaxIter"]):
            self.step()
            self.log.append(self.loginstant())
            # Do logging.
            iter = iter + 1

# need to put this in separate Files
