import numpy as np
import scipy
import scipy.linalg
from func import *
from pyscf import gto, dft, scf, ao2mo
from tdfields import *

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
        self.the_scf  = the_scf_
        self.params = dict()
        self.params["dt"] = 0.02
        self.params["MaxIter"] = 10000000
        self.params["Model"] = "TDDFT"
        self.params["Method"] = "MMUT"

        self.t = 0.0
        self.rho = None # Current MO basis density matrix.
        self.rhoM12 = None # For MMUT step

        #Global variables
        self.n_ao = None
        self.n_mo = None
        self.n_occ = None
        self.X = None


        self.log = []
        self.field = fields(the_scf_, self.params)

        self.initialcondition()
        #self.prop()
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
        n_mo = self.n_mo = n_ao
        n_occ = self.n_occ = int(sum(self.the_scf.mo_occ)/2)

        print "n_ao:", n_ao, "n_mo:", n_mo, "n_occ:", n_occ






        self.ReadParameter()
        self.InitializeLiouvillian()


        return

    def ReadParameter(self):
        '''
        read the file and fill the params dictionary
        '''
        return
    def InitializeLiouvillian(self):
        '''
        Building a Fock Matrix and calculating dipole moment
        '''
        self.FockBuild()



        return

    def FockBuild(self):
        '''
        Using Roothan's equation to build a Fock matrix and initial density matrix
        '''
        n_occ = self.n_occ
        Ne = 2 * n_occ
        err = 100
        # Making rotation matrix X(|AO><MO|)
        H = self.the_scf.get_hcore()
        S = self.the_scf.get_ovlp()
        eigvalH, self.X = scf.hf.eig(H,S)

        P = self.the_scf.make_rdm1()
        Veff = dft.rks.get_veff(self.the_scf, None, P)

        F = H + Veff

        # Roothan's Equation
        # Question: Does Fock matrix needs to be solved in MO basis?
        while (err > 10**-10):
            Fold = F
            eigvalF, C = scipy.linalg.eigh(F, S)
            # C: |BO><MO|
            # P/2 = C*C.t() for Occ orbitals
            Pold = P
            P = 2 * np.dot(C[:,:n_occ],np.transpose(C[:,:n_occ]))
            P = 0.6*Pold + 0.4*P
            P = Ne * P / (TrDot(P, S))
            F = H + dft.rks.get_veff(self.the_scf, None, P)
            err = abs(sum(sum(F-Fold)))
        print "Ne:", TrDot(P, S)


        return

    def TDDDFTstep(self):
        if (self.params["Method"] == "RK4"):
            print "Finish step."
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

    def energy(self):
        return 0.0

    def loginstant(self):
        tore = [self.t]+self.dipole()+self.energy()
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
