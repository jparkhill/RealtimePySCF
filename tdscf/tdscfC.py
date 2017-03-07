import numpy as np
import scipy, os
import scipy.linalg
from func import *
from pyscf import gto, dft, scf, ao2mo
from tdfields import *
from cmath import *
from pyscf import lib
import ctypes

print os.getcwd()
LIBRARYPATH = os.getcwd()+"/lib/tdscf"

def load_library(libname):
    if '1.6' in np.__version__:
        if (sys.platform.startswith('linux') or
            sys.platform.startswith('gnukfreebsd')):
            so_ext = '.so'
        elif sys.platform.startswith('darwin'):
            so_ext = '.dylib'
        elif sys.platform.startswith('win'):
            so_ext = '.dll'
        else:
            raise OSError('Unknown platform')
        libname_so = libname + so_ext
        return ctypes.CDLL(os.path.join(os.path.dirname(__file__), libname_so))
    else:
        _loaderpath = os.path.dirname(LIBRARYPATH)
        return np.ctypeslib.load_library(libname, _loaderpath)

libtdscf = load_library( os.path.expanduser('~') +'/tdscf_pyscf/lib/tdscf/libtdscf')

class tdscfC:
    """
    A general TDSCF object.
    Other types of propagations may inherit from this.

    By default it does
    """
    def __init__(self,the_scf_,prm=None,output = 'log.dat'):
        """
        Args:
            the_scf an SCF object from pyscf (should probably take advantage of complex RKS already in PYSCF)
        Returns:
            Nothing.
        """

        #To be Sorted later
        self.Enuc = the_scf_.e_tot - dft.rks.energy_elec(the_scf_,the_scf_.make_rdm1())[0]
        self.eri3c = None
        self.eri2c = None
        self.n_aux = None
        self.muxo = None
        self.muyo = None
        self.muzo = None
        self.mu0 = None
        #Global numbers
        self.t = 0.0
        self.n_ao = None
        self.n_mo = None
        self.n_occ = None
        self.n_e = None
	self.c0 = None # for ee2
	self.c0dot = None # for ee2
	self.j = None

        #Global Matrices
        self.rho = None # Current MO basis density matrix. (the idempotent (0,1) kind)
        self.rhoM12 = None # For MMUT step
        self.F = None # (AO x AO)
        self.eigs = None # current fock eigenvalues.
        self.S = None # (ao X ao)
        self.C = None # (ao X mo)
        self.X = None # AO => LAO
        self.V = None # LAO x current MO
        self.H = None # (ao X ao)  core hamiltonian.
        self.B = None # for ee2
	self.cia = None # for ee2
	self.ciadot = None # for ee2
        self.log = []

        # Objects
        self.the_scf  = the_scf_
        self.mol = the_scf_.mol
        self.auxmol_set()
        self.params = dict()
        self.initialcondition(prm)
        self.field = fields(the_scf_, self.params)
        self.field.InitializeExpectation(self.rho,self.C)
        self.CField()
        self.prop(output)
        return

    def auxmol_set(self,auxbas = "weigend"):
        auxmol = gto.Mole()
        auxmol.atom = self.mol.atom
        auxmol.basis = auxbas
        auxmol.build()
        mol = self.mol
        nao = self.n_ao = mol.nao_nr()
        naux = self.n_aux = auxmol.nao_nr()

        atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
        eri3c = np.empty((nao,nao,naux))
        pi = 0
        for i in range(mol.nbas):
            pj = 0
            for j in range(mol.nbas):
                pk = 0
                for k in range(mol.nbas, mol.nbas+auxmol.nbas):
                    shls = (i, j, k)
                    buf = gto.getints_by_shell('cint3c2e_sph', shls, atm, bas, env)
                    di, dj, dk = buf.shape
                    eri3c[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
                    pk += dk
                pj += dj
            pi += di

        eri2c = np.empty((naux,naux))
        pk = 0
        for k in range(mol.nbas, mol.nbas+auxmol.nbas):
            pl = 0
            for l in range(mol.nbas, mol.nbas+auxmol.nbas):
                shls = (k, l)
                buf = gto.getints_by_shell('cint2c2e_sph', shls, atm, bas, env)
                dk, dl = buf.shape
                eri2c[pk:pk+dk,pl:pl+dl] = buf
                pl += dl
            pk += dk

        self.eri3c = eri3c
        self.eri2c = eri2c

        RSinv = MatrixPower(eri2c,-0.5)
        self.B = np.einsum('ijp,pq->ijq', eri3c, RSinv) # (AO,AO,n_aux)

        return

    def initialcondition(self,prm):
        print '''
        ===================================
        |  Realtime TDSCF module          |
        ===================================
        | J. Parkhill, T. Nguyen          |
        | J. Koh, J. Herr,  K. Yao        |
        ===================================
        | Refs: 10.1021/acs.jctc.5b00262  |
        |       10.1063/1.4916822         |
        ===================================
        '''
        n_ao = self.n_ao = self.the_scf.make_rdm1().shape[0]
        n_mo = self.n_mo = n_ao # should be fixed.
        n_occ = self.n_occ = int(sum(self.the_scf.mo_occ)/2)
        print "n_ao:", n_ao, "n_mo:", n_mo, "n_occ:", n_occ
        self.ReadParams(prm)
        self.InitializeLiouvillian()
        return

    def ReadParams(self,prm):
        '''
        Read the file and fill the params dictionary
        '''



        self.params["Model"] = "TDDFT" #"TDHF"; the difference of Fock matrix and energy
        self.params["Method"] = "MMUT"#"MMUT"

        self.params["dt"] =  0.02
        self.params["MaxIter"] = 15000

        self.params["ExDir"] = 1.0
        self.params["EyDir"] = 1.0
        self.params["EzDir"] = 1.0
        self.params["FieldAmplitude"] = 0.01
        self.params["FieldFreq"] = 0.9202
        self.params["Tau"] = 0.07
        self.params["tOn"] = 7.0*self.params["Tau"]
        self.params["ApplyImpulse"] = 1
        self.params["ApplyCw"] = 0

        self.params["StatusEvery"] = 5000
        # Here they should be read from disk.
        if(prm != None):
            for line in prm.splitlines():
                s = line.split()
                if len(s) > 1:
                    if s[0] == "MaxIter" or s[0] == str("ApplyImpulse") or s[0] == str("ApplyCw") or s[0] == str("StatusEvery"):
                        self.params[s[0]] = int(s[1])
                    elif s[0] == "Model" or s[0] == "Method":
                        self.params[s[0]] = s[1]
                    else:
                        self.params[s[0]] = float(s[1])

        print "============================="
        print "         Parameters"
        print "============================="
        print "Model:", self.params["Model"]
        print "Method:", self.params["Method"]
        print "dt:", self.params["dt"]
        print "MaxIter:", self.params["MaxIter"]
        print "ExDir:", self.params["ExDir"]
        print "EyDir:", self.params["EyDir"]
        print "EzDir:", self.params["EzDir"]
        print "FieldAmplitude:", self.params["FieldAmplitude"]
        print "FieldFreq:", self.params["FieldFreq"]
        print "Tau:", self.params["Tau"]
        print "tOn:", self.params["tOn"]
        print "ApplyImpulse:", self.params["ApplyImpulse"]
        print "ApplyCw:", self.params["ApplyCw"]
        print "=============================\n\n"

        return

    def InitializeLiouvillian(self):
        '''
        Get an initial Fock matrix.
        '''
        S = self.S = self.the_scf.get_ovlp()
        self.X = MatrixPower(S,-1./2.)
        self.H = self.the_scf.get_hcore()
        self.C = self.X.copy() # Initial set of orthogonal coordinates.
    	if(self.params["Model"] == "TDDFT"):
    	    self.InitFockBuildC()
    	elif(self.params["Model"] == "TDCI") and (self.params["Method"] == "CIS"):
            self.InitFockBuildC2()
        elif(self.params["Model"] == "TDCI") and (self.params["Method"] == "CISD"):
            raise Exception("Unknown Method...")
        self.rho = 0.5*np.diag(self.the_scf.mo_occ).astype(complex)
        self.rhoM12 = self.rho.copy()
        self.get_HJK()
        return


    def get_HJK(self):

        P = TransMat(2*self.rho,self.C,-1)

        naux = self.n_aux
        nao = self.n_ao

        rho = np.einsum('ijp,ij->p', self.eri3c, P)
        rho = np.linalg.solve(self.eri2c, rho)
        J = np.einsum('p,ijp->ij', rho, self.eri3c)

        kpj = np.einsum('ijp,jk->ikp', self.eri3c, P)
        pik = np.linalg.solve(self.eri2c, kpj.reshape(-1,naux).T.conj())
        K = np.einsum('pik,kjp->ij', pik.reshape(naux,nao,nao), self.eri3c)

        self.HJK = self.H + J - 0.5*K


    def InitFockBuildC(self):
        '''
        Transfer H,S,X,B to CPP
        Initialize V matrix and eigs vec in CPP as well
        Make Fockbuild
        Transfer V,C
        '''
        F = np.zeros((self.n_ao,self.n_ao)).astype(complex)
        libtdscf.Initialize(\
        self.H.ctypes.data_as(ctypes.c_void_p),self.S.ctypes.data_as(ctypes.c_void_p),\
        self.X.ctypes.data_as(ctypes.c_void_p),self.B.ctypes.data_as(ctypes.c_void_p),F.ctypes.data_as(ctypes.c_void_p),\
        ctypes.c_int(self.n_ao),ctypes.c_int(self.n_aux),ctypes.c_int(self.n_occ),\
        ctypes.c_double(self.Enuc), ctypes.c_double(self.params["dt"]))

        V = np.zeros((self.n_ao, self.n_ao)).astype(complex)
        C = np.zeros((self.n_ao, self.n_ao)).astype(complex)
        libtdscf.Call(V.ctypes.data_as(ctypes.c_void_p), C.ctypes.data_as(ctypes.c_void_p))
        self.V = V.copy()
        self.C = C.copy()

    def InitFockBuildC2(self):
        '''
        Transfer H,S,X,B to CPP
        Initialize V matrix and eigs vec in CPP as well
        Make Fockbuild
        '''
        F = np.zeros((self.n_ao,self.n_ao)).astype(complex)
        libtdscf.Initialize2(\
        self.H.ctypes.data_as(ctypes.c_void_p),self.S.ctypes.data_as(ctypes.c_void_p),\
        self.X.ctypes.data_as(ctypes.c_void_p),self.B.ctypes.data_as(ctypes.c_void_p),F.ctypes.data_as(ctypes.c_void_p),\
        ctypes.c_int(self.n_ao),ctypes.c_int(self.n_aux),ctypes.c_int(self.n_occ),\
        ctypes.c_double(self.Enuc), ctypes.c_double(self.params["dt"]))

        V = np.zeros((self.n_ao, self.n_ao)).astype(complex)
        C = np.zeros((self.n_ao, self.n_ao)).astype(complex)
        libtdscf.Call(V.ctypes.data_as(ctypes.c_void_p), C.ctypes.data_as(ctypes.c_void_p))
        self.V = V.copy()
        self.C = C.copy()


    def CField(self):

        '''
        setup field components in C
        '''
        mux = 2*self.field.dip_ints[0].astype(complex)
        muy = 2*self.field.dip_ints[1].astype(complex)
        muz = 2*self.field.dip_ints[2].astype(complex)
        nuc = self.field.nuc_dip

        libtdscf.FieldOn(\
        self.rho.ctypes.data_as(ctypes.c_void_p),\
        nuc.ctypes.data_as(ctypes.c_void_p),\
        mux.ctypes.data_as(ctypes.c_void_p),\
        muy.ctypes.data_as(ctypes.c_void_p),\
        muz.ctypes.data_as(ctypes.c_void_p),\
        ctypes.c_double(self.field.pol[0]), ctypes.c_double(self.field.pol[1]), ctypes.c_double(self.field.pol[2]),\
        ctypes.c_double(self.params["FieldAmplitude"]), ctypes.c_double(self.params["FieldFreq"]),\
        ctypes.c_double(self.params["Tau"]), ctypes.c_double(self.params["tOn"]),\
        ctypes.c_int(self.params["ApplyImpulse"]), ctypes.c_int(self.params["ApplyCw"])
        )

        # Bring the field Components
        n = self.n_ao
        muxo = np.zeros((n,n)).astype(complex)
        muyo = np.zeros((n,n)).astype(complex)
        muzo = np.zeros((n,n)).astype(complex)
        mu0 = np.zeros(3)

        libtdscf.InitMu(\
        muxo.ctypes.data_as(ctypes.c_void_p),\
        muyo.ctypes.data_as(ctypes.c_void_p),\
        muzo.ctypes.data_as(ctypes.c_void_p),\
        mu0.ctypes.data_as(ctypes.c_void_p))

        self.muxo = muxo.copy()
        self.muyo = muyo.copy()
        self.muzo = muzo.copy()
        self.mu0 = mu0.copy()

        return



    def TDDFTstep(self,time):

        if (self.params["Method"] == "MMUT"):
            libtdscf.MMUT_step(self.rho.ctypes.data_as(ctypes.c_void_p), self.rhoM12.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(time))
            # Rho and RhoM12 is updated

        elif (self.params["Method"] == "RK4"):
            newrho = self.rho.copy()
            libtdscf.TDTDA_step(newrho.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(time))
            self.rho = newrho.copy()

        else:
            raise Exception("Unknown Method...")

        return



    def CIstep(self,time):
        if(self.params["Method"] == "CIS"):
	    newrho = self.rho.copy()
            libtdscf.CIS_step(cia.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(c0) , ciadot.ctypes.data_as(ctypes.c_void_p), \
	    ctypes.c_double(c0dot), ctypes.c_double(time))
	    self.rho = newrho.copy()
        else:
            raise Exception("Unknown Method...")


    def step(self,time):
        """
        Performs a step
        Updates t, rho, and possibly other things.
        """
        if (self.params["Model"] == "TDDFT"):
            self.TDDFTstep(time)
        elif (self.params["Model"] == "TDCI"):
            self.CIstep(time)
        else:
            raise Exception("Unknown Model...")
        return


    def dipole(self):
        '''
        returns the current Dipole Moment of the molecule
        '''

        dipole = [TrDot(self.rho,self.muxo),TrDot(self.rho,self.muyo),TrDot(self.rho,self.muzo)] - self.mu0
        return dipole.real

    def EnergyC(self, rho):
        '''
        Args:
            rho: MO Density
        Returns:
            Etot: Total Energy (Need to Include Exc if needed)
        '''
        Plao = TransMat(rho,self.V,-1)
        Pao = TransMat(Plao,self.X)
        Etot = TrDot(Pao,self.H).real + TrDot(Pao,self.HJK).real+self.Enuc

        return Etot

    def loginstant(self,iter):
        np.set_printoptions(precision = 7)
        tore = str(self.t)+" "+str(self.dipole().real).rstrip(']').lstrip('[]')+ " " +str(self.EnergyC(self.rho))
        for i in range(self.n_ao):
            tore += " " + str(self.rho[i,i].real)

        # PrintOut
        if iter%self.params["StatusEvery"] ==0:
            print 't:', self.t, "    Energy:",self.EnergyC(self.rho).real
            print('Dipole moment(X, Y, Z, au): %8.5f, %8.5f, %8.5f' %(self.dipole().real[0],self.dipole().real[1],self.dipole().real[2]))
            print "Pop:"
            for i in range(self.n_ao):
                print("%.5f" % self.rho[i,i].real),
            print "\n"
        return tore

    def prop(self,output):
        """
        The main tdscf propagation loop.
        """

        iter = 0
        self.t = 0
        f = open(output,'a')
        print "\n\nPropagation Begins"
        while (iter<self.params["MaxIter"]):
            self.step(self.t)
            f.write(self.loginstant(iter)+"\n")
            # Do logging.
            iter = iter + 1
            self.t = self.t + self.params["dt"]

        f.close()
