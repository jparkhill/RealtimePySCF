import numpy as np
import scipy
import scipy.linalg
from pyscf import gto, dft, scf, ao2mo
from tdfields import *

class tdscf:
    """
    A general TDSCF object.
    Other types of propagations may inherit from this.
    """
    def __init__(self,the_scf_):
        """
        Args:
            the_scf an SCF object from pyscf.
        Returns:
            Nothing.
        """
        self.the_scf  = the_scf_
        self.params = dict()
        self.params["dt"] = 0.02
        self.params["MaxIter"] = 10000000
        self.log = []
        self.field = tdfields(the_scf_, self.params())
        self.prop()
        return

    def prop(self):
        """
        The main tdscf propagation loop.
        """
        iter = 0
        while (iter<MaxIter):
            self.step()
