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
    during the propagation. Including TDTDA, BBGKY1, BBGKY2.
    """
    def __init__(self,the_scf_,prm=None,output = 'log.dat'):
        """
        Args:
            the_scf an SCF object from pyscf (should probably take advantage of complex RKS already in PYSCF)
        Returns:
            Nothing.
        """
    	tdscf.tdscf.__init__(self, the_scf_, prm, output, prop_=False)
        self.rho0 = self.rho.copy()
        self.Bm = None
        self.n = self.n_mo
        self.nso = 2*self.n
        self.Vso = None # Spin orbital integrals for debugging purposes.
        self.rho2 = None # Two body DM in spin-orbital basis.
        if (self.params["BBGKY"] == 2):
            rhoso = self.SOForm(self.rho)
            self.rho2 = self.SeparableR2(rhoso)
        self.MakeBMO()
        self.MakeVi()
        self.BuildSpinOrbitalV()
        self.prop(output)
        return

    def SOForm(self, a_):
        """ Returns the spin-orbital form of a singlet matrix"""
        tmp=np.zeros((2*a_.shape[0],2*a_.shape[1]),dtype=np.complex)
        tmp[:a_.shape[0],:a_.shape[0]] += a_
        tmp[a_.shape[0]:,a_.shape[0]:] += a_
        return tmp

    def SIForm(self, a_):
        """ Returns the spin-integrated form of a singlet matrix"""
        return 0.5*(a_[a_.shape[0]/2:,a_.shape[1]/2:]+a_[:a_.shape[0]/2,:a_.shape[1]/2])

    def SeparableR2(self, a_):
        """ Make a SeparableR2 """
        tmp = np.einsum('pr,qs->pqrs',a_,a_)
        tmp -= np.einsum('qr,ps->pqrs',a_,a_)
        return tmp

    def MakeBMO(self):
        self.Bmo = np.einsum('jip,ik->jkp', np.einsum('ijp,ik->kjp', self.B, self.C), self.C)

    def MakeVi(self):
		self.Vi = np.einsum('pri,qsi->pqrs', self.Bmo,self.Bmo)
		print "Vi: ", self.Vi
		self.d2 = np.array([[j-i for i in self.eigs] for j in self.eigs])
		if (1):
		    # Check this with a quick MP2... works.
		    T=np.zeros((self.n_occ,self.n_occ,self.n_virt,self.n_virt))
		    for i in range(self.n_occ):
		        for j in range(self.n_occ):
		            for a in range(self.n_occ,self.n_mo):
		                for b in range(self.n_occ,self.n_mo):
		                    T[i,j,a-self.n_occ,b-self.n_occ] = (2.0*self.Vi[i,j,a,b]-self.Vi[i,j,b,a])/(self.eigs[a]+self.eigs[b]-self.eigs[i]-self.eigs[j])
		                    print "T*V", i,j,a,b, self.Vi[i,j,a,b], T[i,j,a-self.n_occ,b-self.n_occ]*self.Vi[i,j,a,b]

		    emp2 = np.einsum('ijab,ijab',T,self.Vi[:self.n_occ,:self.n_occ,self.n_occ:,self.n_occ:])
		    print "***EMP2: ", emp2.real

    def BuildSpinOrbitalV(self):
        self.Vso = np.zeros(shape=(self.nso,self.nso,self.nso,self.nso),dtype = np.float)
        n = self.n
        for p in range(self.nso):
            for q in range(self.nso):
                for r in range(self.nso):
                    for s in range(self.nso):
                        s1=p<n; s2=q<n
                        s3=r<n; s4=s<n
                        P=p%n; Q=q%n
                        R=r%n; S=s%n
                        if ((s1!=s3) or (s2!=s4)):
                            continue
                        self.Vso[p,q,r,s] = self.Vi[P,Q,R,S]
        return

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
        if (self.params["BBGKY"] or self.params["TDTDA"] or self.params["TDCIS"]):
            return self.BBGKYstep(time)
        #else:
        #    raise Exception("Why?")
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
        tmp=np.dot((J+K),Rho_) - np.dot(Rho_,(J+K)) # h + this is the fock part.
        RhoDot_ = tmp;
        # OV-> OV and OO and VV
        J = -2.j*np.einsum("bija,ai->bj",self.Vi[:,self.n_occ:,:,:self.n_occ],Rho_[:self.n_occ,self.n_occ:])
        K = 1.j*np.einsum("biaj,ai->bj",self.Vi[:,self.n_occ:,:self.n_occ,:],Rho_[:self.n_occ,self.n_occ:])
        Feff = (J+K)
        Feff[self.n_occ:,:self.n_occ] *= 0.0
        tmp=np.dot((J+K),self.rho0) - np.dot(self.rho0,(J+K)) # h + this is the fock part.
        RhoDot_ += tmp;
        # VO ->
        J = -2.j*np.einsum("bija,ai->bj",self.Vi[:,:self.n_occ,:,self.n_occ:],Rho_[self.n_occ:,:self.n_occ])
        K = 1.j*np.einsum("biaj,ai->bj",self.Vi[:,:self.n_occ,self.n_occ:,:],Rho_[self.n_occ:,:self.n_occ])
        Feff = (J+K)
        Feff[:self.n_occ,self.n_occ:] *= 0.0
        tmp=np.dot((J+K),self.rho0) - np.dot(self.rho0,(J+K)) # h + this is the fock part.
        RhoDot_ += tmp;
        return RhoDot_

    def MakeRho(self,c0,cia):
	""" Should take C0 and Cia as arguments returns the density matrix."""
	Rho = np.identity(n)
	Rho[self.n_e:self.n,self.n_e:self.n] *= 0.0
	RhoHF = Rho
	Rho *= c0.conj()*c0
	Rho[self.n_occ:self.n,:self.n_occ-1] += 1/np.sqrt(2)*c0.conj()*cia
	Rho[:self.n_occ-1,self.n_occ:self.n] += 1/np.sqrt(2)*cia.conj()*c0
	for a in range(self.n_virt):
		for i in range(self.n_occ):
			Rho += cia[i,a].conj()*cia[i,a]*RhoHF

	for a in range(self.n_virt):
		for i in range(self.n_occ):
			for ap in range(self.n_virt):
				Rho[ap+self.n_occ,a+self.n_occ] += 0.5*cia[i,ap].conj()*cia[i,a]

	for a in range(self.n_virt):
		for i in range(self.n_occ):
			for ip in range(self.n_occ):
				Rho[i,ip] -= 0.5*cia[i,a]*cia[ip,a].conj()

	return Rho
	print "Trace Rho: ", np.trace(Rho)
	print "c0^2: ", c0.conj()*c0
	print "cia^2: ", sum(cia.conj()%cia)

    def CISDOT(self, cia ,c0, ciadot, c0dot, tnow):
	# Make Vi & hmu, Check Vi
        # c0dot
        for a in range(self.n_virt):
            for i in range(self.n_occ):
                c0dot += j * cia[a, i] * hmu[i,no+a]

        # first term of ciadot
        for a in range(self.n_virt):
            for i in range(self.n_occ):
                ciadot[i,a] += - j * (self.eigs[no + a]-self.eigs[i]) * cia[i,a]

        # second term of ciadot
        for a in range(self.n_virt):
            for i in range(self.n_occ):
                for ap in range(self.n_virt):
                    for ip in range(self.n_occ):
                        ciadot[i,a] += - j * cia[ip,ap] * (2 * Vi[(a+no) * n3 + ip * n2 + i * n + (ap+no)] - Vi[(a+no) * n3 + ip * n2 + (ap+no) * n + i])

        # last term of ciadot
        for a in range(self.n_virt):
            for i in range(self.n_occ):
                ciadot[i,a] += j * C0 * hmu[i,no+a]

        for a in range(self.n_virt):
            for i in range(self.n_occ):
                for ap in range(self.n_virt):
                    ciadot[i,a] += j/np.sqrt(2)* cia[i,ap] * hmu[no+a,no+ap]

        for a in range(self.n_virt):
            for i in range(self.n_occ):
                for ip in range(self.n_occ):
                    ciadot[i,a] += -j/np.sqrt(2)* cia[ip,a] * hmu[i,ip]

	return

    def Transform2(self, r2_, u_ ):
        """ Perform a two particle unitary transformation. """
        u_so = self.SOForm(u_)
        tmp = np.einsum("tqrs,pt->pqrs", r2_, u_so)
        tmp = np.einsum("ptrs,qt->pqrs", tmp, u_so)
        tmp = np.einsum("pqts,tr->pqrs", tmp, u_so.T.conj())
        tmp = np.einsum("pqrt,ts->pqrs", tmp, u_so.T.conj())
        return tmp

	def CISRK4step(self, tnow):
		if (self.params["Print"]):
			print "CIS step"

        k1 = np.zeros(self.n_virt,self.n_occ)
	k2= np.zeros(self.n_virt,self.n_occ)
	k3 = np.zeros(self.n_virt,self.n_occ)
	k4= np.zeros(self.n_virt,self.n_occ)

        k01 = 0.0; k02 = 0.0; k03 = 0.0; k04 = 0.0

        CISDOT(cia, c0, k1, k01, tnow)
        CISDOT(cia+k1*dt/2.0, c0+k01*dt/2.0, k2, k02, tnow+dt/2.0)
        CISDOT(cia+k2*dt/2.0, c0+k02*dt/2.0, k3, k03, tnow+dt/2.0)
        CISDOT(cia+k3*dt, c0+k03*dt, k4, k04, tnow+dt)

        cia += dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        c0 += dt/6.0 * (k01 + 2*k02 + 2*k03 + k04)

        sum1 = c0*c0.conj().real + np.cumsum(cia%cia.conj())
        cia /= np.sqrt(sum1)
        c0 /= np.sqrt(sum1)

        dRho_ = MakeRho() - dRho_
        cout << endl << tnow << endl
        print "dRho", dRho_

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

        if (self.params["BBGKY"]==2):
            # in this case derive rho from rho2.
            rhoso = np.einsum('prqr->pq', self.rho2)
            self.rho = self.SIForm(rhoso)

        rhoHalfStepped = TransMat(self.rho,U,-1)
        newrho = rhoHalfStepped.copy()
        # one body parts.
        if (self.params["BBGKY"] != 2):
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
        else:
            #propagate rho and r2 together.
            r2 = self.Transform2(self.rho2,U)

            k1_2 = self.bbgky2(r2)
            v2_2 = (0.5 * self.params["dt"])*k1_2
            v2_2 += r2
            k2_2 = self.bbgky2(v2_2)
            v3_2 = (0.5 * self.params["dt"])*k2_2
            v3_2 += r2
            k3_2 = self.bbgky2(v3_2)
            v4_2 = (1.0 * self.params["dt"])*k3_2
            v4_2 += r2
            k4_2 = self.bbgky2(v4_2)
            r2 += self.params["dt"]/6.0 * (k1_2 + 2.0*k2_2 + 2.0*k3_2 + k4_2)
            self.rho2 = self.Transform2(r2, U)

            # Get the traces of r2...
            # which should be the number of electron pairs.
            # print "R2 traces: ",np.einsum("pqpq",self.rho2)

            self.rho = self.SIForm(np.einsum("prqr->pq",self.rho2)/(self.n_e - 1.0))
            self.rhoM12 = self.rho.copy()
            return
        self.rho = TransMat(newrho, U, -1)
        self.rhoM12 = self.rho.copy()
        return

    def bbgky1(self,rho_):
        if (self.params["TDTDA"]):
            return self.rhoDotTDTDA(rho_)
	elif (self.params["TDCIS"]):
	    return self.rhoDotCIS(rho_)
	elif (self.params["TDCISD"]):
	    return self.rhoDotCISD(rho_)
        else:
            rhoso = self.SOForm(rho_)
            r2 = self.SeparableR2(rhoso)
            drhoso = np.einsum("prst,stqr->pq", self.Vso, r2)
            drhoso -= np.einsum("prst,stqr->pq", r2, self.Vso)
            return -1.j*self.SIForm(drhoso)

    def bbgky2(self, rho2_):
        """ The second equation of the BBGKY heirarchy.  """
        #r2d[tn3*t+tn2*u+tn*r+s] += -j*(vso[tn3*t+tn2*u+tn*p+q]*R2[tn3*p+tn2*q+tn*r+s]);
        #r2d[tn3*p+tn2*q+tn*t+u] -= -j*(R2[tn3*p+tn2*q+tn*r+s]*vso[tn3*r+tn2*s+tn*t+u]);
        dr2 = np.einsum("tupq,pqrs->turs",self.Vso,rho2_)
        dr2 -= np.einsum("pqrs,rstu->pqtu",rho2_,self.Vso)
        return -1.j*dr2
