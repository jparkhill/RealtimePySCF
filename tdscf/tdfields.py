import numpy as np
import scipy
import scipy.linalg
from pyscf import gto, dft, scf, ao2mo

class fields:
    """
    A class which manages field perturbations. Mirrors TCL_FieldMatrices
    """
    def __init__(self,the_scf_, params_):
        self.Mux = None
        self.Muy = None
        self.Muz = None
        self.Generate(the_scf_)
        return

    def Generate(self,the_scf):
        """
        Performs the required PYSCF calls to generate the AO basis dipole matrices.
        """
        return

    def ApplyField(self, a_mat, time):
        return

    def Expectation(self,rho_, C_):
        """
        Args:
            rho_: current AO density.
            C_: current AO=> Mo Transformation.
        Returns:
            [<Mux>,<Muy>,<Muz>]
        """
        return
