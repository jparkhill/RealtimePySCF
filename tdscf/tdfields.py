import numpy as np
import scipy
import scipy.linalg
from pyscf import gto, dft, scf, ao2mo


class fields:
    """
    A class which manages field perturbations. Mirrors TCL_FieldMatrices
    """
    def __init__(self,the_scf_, params_):
        return
