import numpy as np
import scipy
import scipy.linalg
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

        self.log = []
        self.field = fields(the_scf_, self.params)

        self.initialcondition()
        self.prop()
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
