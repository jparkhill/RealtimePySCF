import numpy as np
import scipy
import scipy.linalg
from pyscf import gto, dft, scf, ao2mo

class fields:
    """
    A class which manages field perturbations. Mirrors TCL_FieldMatrices.h
    """
    def __init__(self,the_scf_, params_):
        self.dip_ints = None # AO dipole integrals.
        self.Generate(the_scf_)
        self.fieldAmplitude = params_["FieldAmplitude"]
        self.tOn = params_["tOn"]
        self.Tau = params_["Tau"]
        self.FieldFreq = params_["FieldFreq"]
        self.pol = np.array([params_["ExDir"],params_["EyDir"]]),params_["EzDir"]])])
        self.pol0 = None
        return

    def Generate(self,the_scf):
        """
        Performs the required PYSCF calls to generate the AO basis dipole matrices.
        """
        self.dip_ints = the_scf.mol.intor('cint1e_r_sph', comp=3) # ao,ao,component.
        return

    def ImpulseAmp(self,time):
        amp = self.fieldAmplitude*np.sin(self.FieldFreq*time)*(1.0/sqrt(2.0*3.1415*self.Tau))*np.exp(-1.0*np.power(time-t0,2.0)/(2.0*self.Tau*self.Tau));
        IsOn = False
        if (np.abs(amp)>pow(10.0.0,-9.0)):
            IsOn = True
        return amp,IsOn

    def InitializeExpectation(self,rho0_, C_):
        self.pol0 = self.Expectation(rho0_,C_)

    def ApplyField(self, a_mat, time):
        """
        Args:
            a_mat: an AO matrix to which the field is added.
            time: current time.
        Returns:
            a_mat + dipole field at this time.
            IsOn
        """
        mpol, IsOn = self.pol*self.ImpulseAmp(time)
        if (IsOn):
            return a_mat + np.einsum("ijk,k->ij",self.dip_ints,mpol), True
        else:
            return a_mat, False

    def ApplyField(self, a_mat, c_mat, time):
        """
        Args:
            a_mat: an AO matrix to which the field is added.
            c_mat: a AO=>MO coefficient matrix.
            time: current time.
        Returns:
            a_mat + dipole field at this time.
            IsOn
        """
        mpol, IsOn = self.pol*self.ImpulseAmp(time)
        if (IsOn):
            return a_mat + np.einsum("ijk,k->ij",self.dip_ints,mpol), True
        else :
            return a_mat, False

    def Expectation(self, rho_, C_):
        """
        Args:
            rho_: current AO density.
            C_: current AO=> Mo Transformation. (ao X mo)
        Returns:
            [<Mux>,<Muy>,<Muz>]
        """
        # At this point convert both into MO and then calculate the dipole...
        rhoMO = np.dot(np.dot(C_.T,rho_),C_)
        muMO1 = np.einsum("mnk,mi->ink",self.dip_ints,C_)
        muMO = np.einsum("ink,nj->ijk",muMO1,C_)
        if (self.pol0 != None):
            return np.einsum("ij,ji",rhoMO,muMO) - self.pol0
        else:
            return np.einsum("ij,ji",rhoMO,muMO)
