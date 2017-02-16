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

class tdcis(TDSCF):
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
    	TDSCF.__init__(self, the_scf_, prm, output, prop_=False)
        self.Bm = None
        self.MakeBMO()
        self.MakeVi()
        return

    def MakeBMO(self):
        self.Bmo = np.einsum('jip,ik->jkp', np.einsum('ijp,ik->kjp', self.B, self.C), self.C)

    def MakeV(self):
        self.Vi = np.einsum('pqi,rsi->pqrs', self.Bmo,self.Bmo)

    def Split_RK4_Step(self, w, v , oldrho , tnow, dt ,IsOn):
        Ud = np.exp(w*(-0.5j)*dt);
        U = TransMat(np.diag(Ud),v,-1)
        RhoHalfStepped = TransMat(oldrho,U,-1)
        k1 = self.RhoDot( RhoHalfStepped, tnow, IsOn);
        v2 = (dt/2.0) * k1;
        v2 += RhoHalfStepped;
        k2 = self.RhoDot(  v2, tnow+(dt/2.0), IsOn);
        v3 = (dt/2.0) * k2;
        v3 += RhoHalfStepped;
        k3 = self.RhoDot(  v3, tnow+(dt/2.0), IsOn);
        v4 = (dt) * k3;
        v4 += RhoHalfStepped;
        k4 = self.RhoDot(  v4,tnow+dt,IsOn);
        newrho = RhoHalfStepped;
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
            nocs, nos = np.linalg.eig(self.Rho)
            print "Noocs: ", nocs

        # Make the exponential propagator.
        Fmo = np.diag(self.eigs).astype(complex)
        FmoPlusField, IsOn = self.field.ApplyField(Fmo,self.C, time)
        w,v = scipy.linalg.eig(FmoPlusField)

        # Full step RhoM12 to make new RhoM12.
        NewRhoM12 = Split_RK4_Step(w, v, self.RhoM12, time, dt)
        NewRho = Split_RK4_Step(w, v, NewRhoM12, time, dt/2.)
        self.rho = 0.5*(NewRho+(NewRho.T.conj()));
        self.rhoM12 = 0.5*(NewRhoM12+(NewRhoM12.T.conj()))

    def RhoDot(self, Rho_, time, IsOn):
        if (self.params["TDCIS"]):
            return RhoDotCIS(Rho_)
        elif (self.params["TDCISD"]):
            return RhoDotCISD(Rho_)
        elif (self.params["Corr"]):
            return RhoDotCorr(Rho_, RhoDot_, time)
        else:
            raise Exception("Unknown Rhodot.")

    def RhoDotCIS(self, Rho_):
        """
        The bare fock parts of the EOM should already be done.
        for (int i=0; i<no; ++i)
        {
            for (int a=no; a<n; ++a)
                if (abs(Rho_[a*n+i]) > 0.0000000001)
                    for (int ap=no; ap<n; ++ap)
                        for (int ip=0; ip<no; ++ip)
                        {
                            tmp[ap*n+ip] += j*(2.0*Vi[ap*n3+i*n2+ip*n+a]-Vi[ap*n3+i*n2+a*n+ip])*Rho_[a*n+i];
                        }
        }
        RhoDot_ += tmp+tmp.t();
        """
        tmp = 2.j*np.einsum("bija,ai->bj",self.Vi,Rho_)
        tmp -= j*np.einsum("biaj,ai->bj",self.Vi,Rho_)
        return tmp+tmp.T.conj()

    def BBGKYstep(self):
        """
        Propagation is done in the fock basis.
        """
        if (self.params["Print"])
            print "BBGKY step"

        # Make the exponential propagator of HCore for fock and dipole parts.
        hmu = TransMat(self.H,self.C)
        hmu, IsOn = self.field.ApplyField(hmu, self.C, time)
        w,v = scipy.linalg.eig(hmu)
        Ud = np.exp(w*(-0.5j)*dt);
        U = TransMat(np.diag(Ud),v,-1)
        RhoHalfStepped = TransMat(self.Rho,U)
"""
        BuildSpinOrbitalV();

        cx_mat newrho(Rho_); newrho.zeros();
        // one body step.
        {
            arma::cx_mat k1(newrho),k2(newrho),k3(newrho),k4(newrho),v2(newrho),v3(newrho),v4(newrho);
            k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
            v2.zeros(); v3.zeros(); v4.zeros();

            bbgky1( RhoHalfStepped, k1,tnow);
            v2 = (dt/2.0) * k1;
            v2 += RhoHalfStepped;

            bbgky1(  v2, k2,tnow+(dt/2.0));
            v3 = (dt/2.0) * k2;
            v3 += RhoHalfStepped;

            bbgky1(  v3, k3,tnow+(dt/2.0));
            v4 = (dt) * k3;
            v4 += RhoHalfStepped;

            bbgky1(  v4, k4,tnow+dt);
            newrho = RhoHalfStepped;
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

        Rho_ = U*newrho*U.t();
        RhoM12_ = U*newrho*U.t();
        //   TracedownConsistency(Rho_,r2); // Ensures the separable part = Rho at all times.

        if (!(params["BBGKY1"]) && !(params["TDTDA"]))
            return CorrelatedEnergy(r2);
        else
        {
            return MeanFieldEnergy(Rho_);
        }
    }
"""
