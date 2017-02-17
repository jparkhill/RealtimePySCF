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
        if (self.params["BBGKY"]):
            return BBGKYstep()

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
            return rhoDotCIS(rho_)
        elif (self.params["TDCISD"]):
            return rhoDotCISD(rho_)
        elif (self.params["Corr"]):
            return rhoDotCorr(rho_, rhoDot_, time)
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
        tmp = 2.j*np.einsum("bija,ai->bj",self.Vi,rho_)
        tmp -= j*np.einsum("biaj,ai->bj",self.Vi,rho_)
        return tmp+tmp.T.conj()

    def BBGKYstep(self):
        """
        Propagation is done in the fock basis.
        """
        if (self.params["Print"]):
            print "BBGKY step"

        # Make the exponential propagator of HCore for fock and dipole parts.
        hmu = TransMat(self.H,self.C)
        hmu, IsOn = self.field.ApplyField(hmu, self.C, time)
        w,v = scipy.linalg.eig(hmu)
        Ud = np.exp(w*(-0.5j)*dt);
        U = TransMat(np.diag(Ud),v,-1)
        rhoHalfStepped = TransMat(self.rho,U)
"""
        BuildSpinOrbitalV();

        cx_mat newrho(rho_); newrho.zeros();
        // one body step.
        {
            arma::cx_mat k1(newrho),k2(newrho),k3(newrho),k4(newrho),v2(newrho),v3(newrho),v4(newrho);
            k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
            v2.zeros(); v3.zeros(); v4.zeros();

            bbgky1( rhoHalfStepped, k1,tnow);
            v2 = (dt/2.0) * k1;
            v2 += rhoHalfStepped;

            bbgky1(  v2, k2,tnow+(dt/2.0));
            v3 = (dt/2.0) * k2;
            v3 += rhoHalfStepped;

            bbgky1(  v3, k3,tnow+(dt/2.0));
            v4 = (dt) * k3;
            v4 += rhoHalfStepped;

            bbgky1(  v4, k4,tnow+dt);
            newrho = rhoHalfStepped;
            newrho += dt*(1.0/6.0)*k1;
            newrho += dt*(2.0/6.0)*k2;
            newrho += dt*(2.0/6.0)*k3;
            newrho += dt*(1.0/6.0)*k4;
        }
        // Two-body step.
        if (!params["BBGKY1"])
        {
            Transform2(r2,U);

            arma::cx_mat k1(r2),k2(r2),k3(r2),k4(r2),v2(r2),v3(r2),v4(r2);
            k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
            v2.zeros(); v3.zeros(); v4.zeros();

            bbgky2( r2, k1,tnow);
            v2 = (dt/2.0) * k1;
            v2 += r2;
            bbgky2(  v2, k2,tnow+(dt/2.0));
            v3 = (dt/2.0) * k2;
            v3 += r2;
            bbgky2(  v3, k3,tnow+(dt/2.0));
            v4 = (dt) * k3;
            v4 += r2;
            bbgky2(  v4, k4,tnow+dt);
            r2 += dt*(1.0/6.0)*k1;
            r2 += dt*(2.0/6.0)*k2;
            r2 += dt*(2.0/6.0)*k3;
            r2 += dt*(1.0/6.0)*k4;

            Transform2(r2,U);
        }

        rho_ = U*newrho*U.t();
        rhoM12_ = U*newrho*U.t();
        //   TracedownConsistency(rho_,r2); // Ensures the separable part = rho at all times.

        if (!(params["BBGKY1"]) && !(params["TDTDA"]))
            return CorrelatedEnergy(r2);
        else
        {
            return MeanFieldEnergy(rho_);
        }
    }
"""
