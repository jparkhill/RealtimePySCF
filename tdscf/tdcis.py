import numpy as np
import scipy
import scipy.linalg
from func import *
from pyscf import gto, dft, scf, ao2mo
from tdfields import *
from cmath import *
from pyscf import lib
import ctypes
import tdscf
#libtdscf = lib.load_library('libtdscf')

FsPerAu = 0.0241888

class tdcis(tdscf.tdscf):
    """
    TDCIS handles propagations which do not rotate the orbitals.
    during the propagation.
    """
    def __init__(self,the_scf_,prm=None,output = 'log.dat'):
        """
        Args:
            the_scf an SCF object from pyscf (should probably take advantage of complex RKS already in PYSCF)
        Returns:
            Nothing.
        """
    	tdscf.tdscf.__init__(self, the_scf_, prm, output, prop_=False)
        self.Bm = None
        self.MakeBMO()
        self.MakeVi()
        self.prop(output)
        return

    def MakeBMO(self):
        self.Bmo = np.einsum('jip,ik->jkp', np.einsum('ijp,ik->kjp', self.B, self.C), self.C)

    def MakeVi(self):
        self.Vi = np.einsum('pqi,rsi->pqrs', self.Bmo,self.Bmo)
        self.d2 = np.array([[j-i for i in self.eigs] for j in self.eigs])
        print self.d2
        # Check this with a quick MP2...
        T=np.zeros((self.n_occ,self.n_occ,self.n_virt,self.n_virt))
        for i in range(self.n_occ):
            for j in range(self.n_occ):
                for a in range(self.n_occ,self.n_mo):
                    for b in range(self.n_occ,self.n_mo):
                        T[i,j,a-self.n_occ,b-self.n_occ] = (2.0*self.Vi[i,j,a,b]-self.Vi[i,j,b,a])/(self.eigs[a]+self.eigs[b]-self.eigs[i]-self.eigs[j])
                        print "T*V", i,j,a,b, self.Vi[i,j,a,b], T[i,j,a-self.n_occ,b-self.n_occ]*self.Vi[i,j,a,b]

        emp2 = np.einsum('ijab,ijab',T,self.Vi[:self.n_occ,:self.n_occ,self.n_occ:,self.n_occ:])
        print "EMP2******", emp2

    def Split_RK4_Step(self, w, v , oldrho , tnow, dt):
        IsOn = False
        Ud = np.exp(w*(-0.5j)*dt);
        U = TransMat(np.diag(Ud),v,-1)
        rhoHalfStepped = TransMat(oldrho,U,-1)
        k1 = self.rhoDot( rhoHalfStepped, tnow, IsOn);
        v2 = (dt/2.0) * k1;
        v2 += rhoHalfStepped;
        k2 = self.rhoDot(  v2, tnow+(dt/2.0), IsOn);
        v3 = (dt/2.0) * k2;
        v3 += rhoHalfStepped;
        k3 = self.rhoDot(  v3, tnow+(dt/2.0), IsOn);
        v4 = (dt) * k3;
        v4 += rhoHalfStepped;
        k4 = self.rhoDot(  v4,tnow+dt,IsOn);
        newrho = rhoHalfStepped;
        newrho += dt*(1.0/6.0)*k1;
        newrho += dt*(2.0/6.0)*k2;
        newrho += dt*(2.0/6.0)*k3;
        newrho += dt*(1.0/6.0)*k4;
        newrho = TransMat(newrho,U,-1)
        return newrho

    def step(self,time):
        """
        Direct port of our gen_scfman/TCL_EE2.h::step()
        """
        if (self.params["Model"]=="BBGKY"):
            return self.BBGKYstep(time)

        if (self.params["Print"]>0.0):
            nocs, nos = np.linalg.eig(self.rho)
            print "Noocs: ", nocs

        # Make the exponential propagator.
        Fmo = np.diag(self.eigs).astype(complex)
        FmoPlusField, IsOn = self.field.ApplyField(Fmo,self.C, time)
        w,v = scipy.linalg.eig(FmoPlusField)

        # Full step rhoM12 to make new rhoM12.
        NewrhoM12 = self.Split_RK4_Step(w, v, self.rhoM12, time, self.params["dt"])
        Newrho = self.Split_RK4_Step(w, v, NewrhoM12, time, self.params["dt"]/2.)
        self.rho = 0.5*(Newrho+(Newrho.T.conj()));
        self.rhoM12 = 0.5*(NewrhoM12+(NewrhoM12.T.conj()))

    def rhoDot(self, rho_, time, IsOn):
        if (self.params["TDCIS"]):
            return self.rhoDotCIS(rho_)
        elif (self.params["TDCISD"]):
            return self.rhoDotCISD(rho_)
        elif (self.params["Corr"]):
            return self.rhoDotCorr(rho_, rhoDot_, time)
        else:
            raise Exception("Unknown rhodot.")

    def rhoDotCIS(self, rho_):
        """
        The bare fock parts of the EOM should already be done.
        for (int i=0; i<no; ++i)
        {
            for (int a=no; a<n; ++a)
                if (abs(rho_[a*n+i]) > 0.0000000001)
                    for (int ap=no; ap<n; ++ap)
                        for (int ip=0; ip<no; ++ip)
                        {
                            tmp[ap*n+ip] += j*(2.0*Vi[ap*n3+i*n2+ip*n+a]-Vi[ap*n3+i*n2+a*n+ip])*rho_[a*n+i];
                        }
        }
        rhoDot_ += tmp+tmp.t();
        """
        #print "EH binding energy: ", self.Vi[self.n_occ,self.n_occ-1,self.n_occ,self.n_occ-1]
        tmp = 2.j*np.einsum("bija,ai->bj",self.Vi,rho_)
        tmp -= 1.j*np.einsum("biaj,ai->bj",self.Vi,rho_)
        return tmp+tmp.T.conj()

    def rhoDotTDTDA(self, Rho_):
        """
            Time-dependent tamm-dancoff approximation
            depends on current rho_ which is in the MO basis.
            and Rho0 in the same basis.
        """

        J = -2.j*np.einsum("bija,ai->bj",self.Vi,self.rho0)
        K = 1.j*np.einsum("biaj,ai->bj",self.Vi,self.rho0)
        tmp=(J+K)*Rho_ - Rho_*(J+K); # h + this is the fock part.
        RhoDot_ = tmp;

        # Eliminate OV-VO couplings.
        if 0:
            J = -2.j*np.einsum("bija,ai->bj",self.Vi,Rho0)
            K = 1.j*np.einsum("biaj,ai->bj",self.Vi,Rho0)
            tmp=(J+K)*Rho_ - Rho_*(J+K); # h + this is the fock part.
            RhoDot_ += tmp;

            J = -2.j*np.einsum("bija,ai->bj",self.Vi,Rho0)
            K = 1.j*np.einsum("biaj,ai->bj",self.Vi,Rho0)
            tmp=(J+K)*Rho_ - Rho_*(J+K); # h + this is the fock part.
            RhoDot_ += tmp;
        return RhoDot_

    def BBGKYstep(self,time):
        """
        Propagation is done in the fock basis.
        """
        if (self.params["Print"]):
            print "BBGKY step"
        # Make the exponential propagator of HCore for fock and dipole parts.
        hmu = TransMat(self.H,self.C)
        hmu, IsOn = self.field.ApplyField(hmu, self.C, time)
        w,v = scipy.linalg.eig(hmu)
        Ud = np.exp(w*(-0.5j)*self.params["dt"]);
        U = TransMat(np.diag(Ud),v,-1)
        rhoHalfStepped = TransMat(self.rho,U)
        newrho = rhoHalfStepped.copy()
        # one body parts.
        if (1):
            k1 = self.bbgky1(rhoHalfStepped)
            v2 = (0.5 * self.params["dt"])*k1
            v2 += rhoHalfStepped
            k2 = self.bbgky1(v2)
            v3 = (0.5 * self.params["dt"])*k2
            v3 += rhoHalfStepped
            k3 = self.bbgky1(v3)
            v4 = (1.0 * self.params["dt"])*k3
            v4 += rhoHalfStepped
            k4 = self.bbgky1(v4)
            newrho += self.params["dt"]/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
        self.rho = TransMat(newrho,U)
        self.rhoM12 = TransMat(newrho,U)

    def bbgky1(self,rho_):
        return self.rhoDotTDTDA(rho_)
        # Here I should have code for the normal Bbgky1 ported out of qchem.
