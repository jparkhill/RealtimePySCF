import numpy as np
from cmath import *
from func import *
import scipy
import scipy.linalg
from pyscf import gto, dft, scf, ao2mo

class fields:
    """
    A class which manages field perturbations. Mirrors TCL_FieldMatrices.h
    """
    def __init__(self,the_scf_, params_):
        self.dip_ints = None # AO dipole integrals.
        self.nuc_dip = None
        self.dip_mo = None # Nuclear dipole (AO)
        self.Generate(the_scf_)
        self.fieldAmplitude = params_["FieldAmplitude"]
        self.tOn = params_["tOn"]
        self.Tau = params_["Tau"]
        self.FieldFreq = params_["FieldFreq"]
        self.pol = np.array([params_["ExDir"],params_["EyDir"],params_["EzDir"]])
        self.pol0 = None
        return

    def Generate(self,the_scf):
        """
        Performs the required PYSCF calls to generate the AO basis dipole matrices.
        """
        self.dip_ints = the_scf.mol.intor('cint1e_r_sph', comp=3) # component,ao,ao.
        #print "A dipole matrices\n",self.dip_ints
        charges = the_scf.mol.atom_charges()
        coords  = the_scf.mol.atom_coords()
        self.nuc_dip = np.einsum('i,ix->x', charges, coords)
        return

    def Update(self,c_mat):
        '''
        Args:
            c_mat: Transformation matrix (AOx??)
        Updates dip_mo to (?? x ??)
        '''
        self.dip_mo = self.dip_ints.astype(complex).copy()
        self.dip_mo[0] = TransMat(self.dip_ints[0],c_mat,-1)
        self.dip_mo[1] = TransMat(self.dip_ints[1],c_mat,-1)
        self.dip_mo[2] = TransMat(self.dip_ints[2],c_mat,-1)
        return

    def ImpulseAmp(self,time):
        amp = self.fieldAmplitude*np.sin(self.FieldFreq*time)*(1.0/sqrt(2.0*3.1415*self.Tau*self.Tau))*np.exp(-1.0*np.power(time-self.tOn,2.0)/(2.0*self.Tau*self.Tau));
        IsOn = False
        if (np.abs(amp)>pow(10.0,-9.0)):
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
        amp, IsOn = self.ImpulseAmp(time)
        mpol = self.pol * amp
        if (IsOn):
            print "Field on"
            return a_mat + 2.0*np.einsum("kij,k->ij",self.dip_ints,mpol), True
        else:
            return a_mat, False

    def ApplyField(self, a_mat, c_mat, time):
        """
        Args:
            a_mat: an MO matrix to which the field is added.
            c_mat: a AO=>MO coefficient matrix.
            time: current time.
        Returns:
            a_mat + dipole field at this time.
            IsOn
        """
        amp, IsOn = self.ImpulseAmp(time)
        mpol = self.pol * amp
        if (IsOn):
            return a_mat + 2.0*TransMat(np.einsum("kij,k->ij",self.dip_ints,mpol),c_mat), True
        else :
            return a_mat, False

    def Expectation(self, rho_, C_):
        """
        Args:
            rho_: current MO density.
            C_: current AO=> Mo Transformation. (ao X mo)
        Returns:
            [<Mux>,<Muy>,<Muz>]
        """
        # At this point convert both into MO and then calculate the dipole...
        rhoAO = TransMat(rho_,C_,-1)
        e_dip = np.einsum('xij,ji->x', self.dip_ints, rhoAO)
        mol_dip = e_dip

        #mol_dip = np.einsum('xij,ji->x', self.dip_mo, rho_)
        if (self.pol0 != None):
            #print "Original Dipole\n",e_dip - self.pol0
            #print "MO calculated Dipole\n", np.einsum('xij,ji->x', self.dip_mo, rho_) - self.pol0
            return mol_dip - self.pol0#2.0*np.einsum("ij,jik->k",rhoMO,muMO) - self.pol0


        else:

            return mol_dip#2.0*np.einsum("ij,jik->k",rhoMO,muMO)
