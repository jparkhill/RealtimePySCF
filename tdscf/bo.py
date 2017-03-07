import numpy as np
import scipy, time
import scipy.linalg
from func import *
from pyscf import gto, dft, scf, ao2mo
from tdfields import *
import ctypes

FsPerAu = 0.0241888

class BORHF():
    def __init__(self, atom, basis, n, xc = None, prm = None, output = 'log.dat', AA = True):
        """
        Make a BlockOrthogonalizedRHF
        Args: mol1_, mol2_ fragments to make a block orthogonal SCF from.
        """

        self.AA = AA
        self.adiis = None
        self.JKtilde = None
        self.basis = basis
        self.m = []
        self.scf = []
        self.hyb = [0,0,0]
        # objects
        self.m1, self.m2, self.m3 = self.MixedBasisMol(atom,basis,n)
        self.hf1, self.hf2, self.hf3 = self.MixedTheory(xc)
        # Global Variables
        self.AOExc = None
        self.BOExc = None
        self.EBO = None
        self.nA = self.m1.nao_nr()
        self.n = self.m3.nao_nr()
        self.n_ao = np.zeros(3)
        self.n_mo = None
        self.n_aux = np.zeros(3)
        self.n_occ = int(sum(self.hf3.mo_occ)/2)
        self.Enuc = self.hf3.energy_nuc()
        self.Exc = np.zeros(3)
        # Global Matrices
        self.C = None
        self.eri3c = []
        self.eri2c = []
        self.H = self.hf3.get_hcore()
        self.S = self.hf3.get_ovlp() # AOxAO
        self.U = BOprojector(self.m1,self.m3) #(AO->BO)
        self.Htilde = TransMat(self.H,self.U)
        self.Stilde = TransMat(self.S,self.U)
        self.X = MatrixPower(self.S,-1./2.) # AO->LAO
        self.Xtilde = MatrixPower(self.Stilde,-1./2.)
        self.O = None #(MO->BO) = C * U
        self.Vtilde = None

        # Propagation Steps
        #print self.hyb
        self.B0 = None # nA,nA, n_aux0
        self.B1 = None # n, n, n_aux1
        self.auxmol_set()
        self.CBOsetup()
        self.params = dict()
        self.initialcondition(prm)
        self.field = fields(self.hf3, self.params)
        self.field.InitializeExpectation(self.rho, self.C, self.nA)
        #quit()
        start = time.time()
        self.prop(output)
        end = time.time()
        print "Propagation time:", end - start
        return

    def CBOsetup(self):
        print "============================="
        libtdscf.SetupBO(\
        ctypes.c_int(int(self.nA)),ctypes.c_int(int(self.n_ao[2])),ctypes.c_int(int(self.n_occ)),ctypes.c_int(int(self.n_aux[2])),ctypes.c_int(int(self.n_aux[0])),\
        self.U.ctypes.data_as(ctypes.c_void_p), \
        self.B0.ctypes.data_as(ctypes.c_void_p), self.B1.ctypes.data_as(ctypes.c_void_p))

    def MixedBasisMol(self, atm, bas, n):
        # For mixed Basis of AA and BB
        p = 0
        n1 = 0
        n2 = 0
        atm1 = ''
        atm2 = ''
        set1 = []
        set2 = []

        for line in atm.splitlines():
            line1 = line.lstrip()
            if len(line1) > 1:
                if p < n:
                    atm1 += "@"+line1+"\n"
                    set1.append("@"+line1[0:1])
                    n1 += 1
                else:
                    atm2 += line1 + "\n"
                    set2.append(line1[0:1])
                    n2 += 1
                p += 1
        atm0 = atm1 + atm2
        bas12 = {}
        for i in range(n1):
            bas12[set1[i]] = bas[0]
        for i in range(n2):
            bas12[set2[i]] = bas[1]

        mol1 = gto.Mole()
        mol1.atom = atm1
        mol1.basis = bas[0]
        mol1.build()

        mol2 = mol1

        mol3 = gto.Mole()
        mol3.atom = atm0
        mol3.basis = bas12
        mol3.build()


        return mol1,mol2,mol3

    def MixedTheory(self,xc):
        # Generate Mixed HF objects
        print "\n================"
        print "=   AA Block   ="
        print "================"
        print "Basis:", self.basis[0]
        print "Theory:",xc[0]
        m1 = dft.rks.RKS(self.m1)
        m1.xc = xc[0]
        m1.grids.level = 1
        m1.kernel()
        self.scf.append(m1)
        self.m.append(self.m1)
        self.hyb[0] = m1._numint.hybrid_coeff(m1.xc, spin=(m1.mol.spin>0)+1)
        print "Basis:", self.basis[0]
        print "Theory:",xc[1]
        m2 = dft.rks.RKS(self.m2)
        m2.xc = xc[1]
        m2.grids.level = 1
        m2.kernel()
        self.scf.append(m2)
        self.m.append(self.m2)
        self.hyb[1] = m2._numint.hybrid_coeff(m2.xc, spin=(m2.mol.spin>0)+1)
        print "\n================="
        print "=  Whole Block  ="
        print "================="
        print "Basis:", self.basis[1]
        print "Theory:",xc[1]
        #m3 = scf.hf.RHF(self.m3)
        m3 = dft.rks.RKS(self.m3)
        m3.xc = xc[1]
        m3.grids.level = 1
        m3.kernel()
        self.scf.append(m3)
        self.m.append(self.m3)
        self.hyb[2] = m3._numint.hybrid_coeff(m3.xc, spin=(m3.mol.spin>0)+1)

        #hf4 = dft.rks.RKS(self.m3)
        #hf4.xc = xc[0]
        #self.hf4 = hf4


        return m1,m2,m3


    def auxmol_set(self,auxbas = "weigend"):
        print "===================="
        print "GENERATING INTEGRALS"
        print "===================="
        auxmol = gto.Mole()
        auxmol.atom = self.hf1.mol.atom
        auxmol.basis = auxbas
        auxmol.build()
        mol = self.hf1.mol
        nao = self.n_ao[0] = mol.nao_nr()
        naux = self.n_aux[0] = auxmol.nao_nr()
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


        self.eri3c.append(eri3c)
        self.eri2c.append(eri2c)
        print "\nAA INT GENERATED"

        self.eri3c.append(eri3c)
        self.eri2c.append(eri2c)

        RSinv = MatrixPower(eri2c,-0.5)
        self.B0 = np.einsum('ijp,pq->ijq', eri3c, RSinv)

        auxmol = mol = nao = naux = None

        auxmol = gto.Mole()
        auxmol.atom = self.hf3.mol.atom
        auxmol.basis = auxbas
        auxmol.build()
        mol = self.hf3.mol
        nao = self.n_ao[2] = mol.nao_nr()
        naux = self.n_aux[2] = auxmol.nao_nr()
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

        # if(self.hyb[0] > 0.01):
        #     eri3cBO = np.zeros((nao,nao,naux))
        #     for i in range(naux):
        #         eri3cBO[:,:,i] = TransMat(eri3c[:,:,i],self.U)
        #     eri3c = eri3cBO.copy()
        print "\nWHOLE INT GENERATED"
        self.eri3c.append(eri3c)
        self.eri2c.append(eri2c)
        RSinv = MatrixPower(eri2c,-0.5)
        self.B1 = np.einsum('ijp,pq->ijq', eri3c, RSinv)
        auxmol = mol = nao = naux = None
        return

    def initialcondition(self,prm):
        n_ao = self.n_ao[2] = self.hf3.mol.nao_nr()#self.hf3.make_rdm1().shape[0]
        n_mo = self.n_mo = n_ao # should be fixed.
        n_occ = self.n_occ = int(sum(self.hf3.mo_occ)/2)
        print "n_ao:", n_ao, "n_mo:", n_mo, "n_occ:", n_occ
        print "nA:",self.nA,"n:", self.n
        self.ReadParams(prm)
        self.InitializeLiouvillian()
        return

    def ReadParams(self,prm):
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
        print "StatusEvery:", self.params["StatusEvery"]
        print "=============================\n\n"
        return

    def InitializeLiouvillian(self):
        self.InitFockBuild()
        self.rho = 0.5*np.diag(self.hf3.mo_occ).astype(complex)
        self.rhoM12 = self.rho.copy()

    def InitFockBuild(self):
        '''
        Using Roothan's equation to build a Fock matrix and initial density matrix
        Returns:
            self consistent density in Block Orthogonalized basis.
        '''
        nA = self.nA
        n = self.n
        n_occ = self.n_occ
        Ne = self.n_e = 2.0 * n_occ
        noc = int(Ne/2)
        err = 100
        it = 0
        dm = self.hf3.make_rdm1()
        Uinv = np.linalg.inv(self.U)
        Pbo = 0.5 * TransMat(dm, Uinv, -1)
        print dm - 2.0 * TransMat(Pbo,self.U,-1)
        print "Ne (AO):", TrDot(dm, self.S)
        print "Ne (BO):", TrDot(Pbo,self.Stilde)

        adiis = self.hf3.DIIS(self.hf3, self.hf3.diis_file)
        adiis.space = self.hf3.diis_space
        adiis.rollback = self.hf3.diis_space_rollback
        self.adiis = adiis

        self.F = self.BO_Fockbuild(Pbo)
        E = self.energy(Pbo)
        print "Initial Energy:",E
        e_conv = 10**-7
        while (err > e_conv):
            # Diagonalize F in the BO basis
            self.eigs, Ctilde = scipy.linalg.eigh(self.F,self.Stilde)
            occs = np.ones(self.n,dtype = np.complex)
            occs[noc:] *= 0.0
            Pmo = np.diag(occs).astype(complex)
            Pbo = TransMat(Pmo,Ctilde,-1)
            print "NE(Lit):", TrDot(Pbo,self.Stilde)
            Eold = E
            self.F = self.BO_Fockbuild(Pbo,it) # in BO Basis
            E = self.energy(Pbo)
            err = abs(E-Eold)
            if (it%1 ==0):
                print "Iteration:", it,"; Energy:",E,"; Error =",err
            it += 1
            if(it == 50):
                print "Lowering Error limit to 10^-6"
                e_conv = 10**-6
            elif(it == 100):
                print "Lowering Error limit to 10^-5"
                e_conv = 10**-5
            elif(it == 200):
                print "SCF Convergence not met"
                break
        self.Ctrev = np.dot(self.Stilde,Ctilde)
        self.C = np.dot(self.U, Ctilde) #|AO><BO|
        Pmo = 0.5*np.diag(self.hf3.mo_occ).astype(complex)
        Pbo = TransMat(Pmo,Ctilde,-1)
        self.Ctilde = Ctilde.copy()
        self.rho = TransMat(Pbo,self.Ctrev)# BO to MO
        self.rhoM12 = TransMat(Pbo,self.Ctrev)
        print "Ne", TrDot(Pbo,self.Stilde), np.trace(self.rho),
        print "Energy:",E
        print "Initial Fock Matrix(MO)\n", TransMat(self.F,Ctilde)
        print self.eigs
        #quit()
        return Pbo


    def BO_Fockbuild(self,P,it = -1):
        """
        Updates self.F given current self.rho (both complex.)
        Fock matrix with HF
        Args:
            P = BO density matrix.
        Returns:
            Fock matrix(BO) . Updates self.Jtilde, Ktilde, JKtilde
        """
        nA = self.nA
        Pt = 2.0*TransMat(P,self.U,-1)
        PtA = 2.0*P[:nA,:nA]
        if self.params["Model"] == "TDHF":
            J,K = self.get_jk(Pt,2)
            self.Jtilde = TransMat(J,self.U) # AO to BO
            self.Ktilde = TransMat(K,self.U) # AO TO BO

            self.JKtilde = self.Jtilde - 0.5* self.Ktilde # combine in BO

            return self.Htilde + 0.5 * (self.JKtilde + self.JKtilde.T.conj())
        elif self.params["Model"] == "TDDFT":
            #Ptb = 2.0 * P
            #self.Jtilde = self.get_j(Ptb,2)
            J = self.get_j(Pt,int(2))
            #self.Jtilde = J
            self.Jtilde = TransMat(J,self.U)
            #self.Jtilde[:nA,:nA] += self.get_j(PtA,0)
            #self.Jtilde[:nA,:nA] -= self.get_j(PtA,1)
            Veff = self.get_vxc(Pt,int(2))
            Veff = TransMat(Veff,self.U)
            Veff[:nA,:nA] += self.get_vxc(PtA,int(0))
            Veff[:nA,:nA] -= self.get_vxc(PtA,int(1))
            if(self.hyb[0] > 0.01):
                self.Ktilde = np.zeros((self.n,self.n)).astype(complex)

                # this step may diverge the fockbuild(confirmed)
                #Ktilde = self.get_k(Pt,int(2))
                #self.Ktilde[:nA,:nA] = Ktilde[:nA,:nA].copy()

                #Ktilde = self.get_k(PtA,int(0))
                Ktilde = self.get_k_c(PtA,int(0))
                self.Ktilde[:nA,:nA] = Ktilde


                Veff[:nA,:nA] += -0.5 * self.hyb[0] * self.Ktilde[:nA,:nA]
            JKtilde = self.Jtilde + Veff
            if self.adiis and it > 0:
                # Kevin: Do the extrapolation with all AO quantities and return the extrapolated fock matrix in BO basis.
                #return TransMat(self.adiis.update(self.S,Pt,self.H + 0.5 * (JKtilde + JKtilde.T.conj())),self.U)
                return self.adiis.update(self.Stilde,2*P,self.Htilde + 0.5 * (JKtilde + JKtilde.T.conj()))
            else:
                return self.Htilde + 0.5 * (JKtilde + JKtilde.T.conj())


    def get_vxc(self,P,mol = 3):
        '''
        Args:
            P: BO density matrix

        Returns:
            Vxc: Exchange and Correlation matrix (AO)
        '''
        Vxc = self.numint_vxc(self.scf[mol]._numint,P,mol)
        return Vxc


    def energy(self,Pbo,IfPrint=False):
        nA = self.nA
        n = self.n
        if self.params["Model"] == "TDHF":
            #print TrDot(TransMat(Pbo,self.U,-1),2*self.H)
            EH = np.trace(np.dot(Pbo,2.0 * self.Htilde))
            EJK = np.trace(np.dot(Pbo,1.0* self.JKtilde))
            #EK = np.trace(np.dot(Pbo,-0.5 * self.K))
            #print "One Electron", EH
            #print "Two Electron", EJK
            #print "Coulomb:", EJ
            #print "Exchange:", EK
            #print "Nuclear Energy:", self.Enuc
            return (self.Enuc + EH + EJK).real
        elif self.params["Model"] == "TDDFT":
            EH = np.trace(np.dot(Pbo,2.0 * self.Htilde))
            EJ = np.trace(np.dot(Pbo,1.0 * self.Jtilde))
            EK = 0
            # Under the assumption that low theory will not use hybrid functional
            if(self.hyb[0] > 0.01):
                EK = self.hyb[0] * TrDot(Pbo, -0.5 * self.Ktilde)
            Exc = self.Exc[2]
            Exch = self.Exc[0]
            Excl = self.Exc[1]
            #print "One Electron", EH
            #print "Coulomb:", EJ
            #print "EXC", Exc
            #print "EXCH", Exch
            #print "EXCL", Excl
            #print "Exchange:", EK
            #print "Nuclear Energy:", self.Enuc
            E = EH + EJ + EK + Exc + Exch - Excl + self.Enuc
            return E.real
        #return (self.Enuc+np.trace(np.dot(Pbo,self.Htilde+self.F))).real

    def numint_vxc(self,ni,P,mol = 3,max_mem = 2000):
        xctype = self.scf[mol]._numint._xc_type(self.scf[mol].xc)
        make_rho, nset, nao = self._gen_rho_evaluator(self.m[mol], P, 1)
        ngrids = len(self.scf[mol].grids.weights)
        non0tab = self.scf[mol]._numint.non0tab
        vmat = np.zeros((nset,nao,nao)).astype(complex)
        excsum = np.zeros(nset)
        #print nset
        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords in ni.block_loop(self.m[mol], self.scf[mol].grids, nao, ao_deriv, max_mem, non0tab):
                rho = make_rho(0, ao, mask, 'LDA')
                exc, vxc = ni.eval_xc(self.scf[mol].xc, rho, 0, 0, 1, None)[:2]
                vrho = vxc[0]
                den = rho * weight
                excsum[0] += (den * exc).sum()
                aow = np.einsum('pi,p->pi', ao, .5*weight*vrho)
                vmat += r_dot_product(ao.T,aow)
                rho = exc = vxc = vrho = aow = None
        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords in ni.block_loop(self.m[mol], self.scf[mol].grids, nao, ao_deriv, max_mem, non0tab):
                ngrid = weight.size
                #ao = ao.astype(complex)
                rho = make_rho(0, ao, mask, 'GGA') # rho should be real
                exc, vxc = ni.eval_xc(self.scf[mol].xc, rho, 0, 0, 1, None)[:2]
                vrho, vsigma = vxc[:2]
                wv = np.empty((4,ngrid))#.astype(complex)
                wv[0]  = weight * vrho * .5
                wv[1:] = rho[1:] * (weight * vsigma * 2)
                aow = np.einsum('npi,np->pi', ao, wv)
                vmat += r_dot_product(ao[0].T,aow) #np.einsum('ij,jk->ik',ao[0].T,aow) #_dot_ao_ao(self.mol, ao[0], aow, nao, ngrid, mask)
                den = rho[0] * weight
                excsum[0] += (den * exc).sum()
                rho = exc = vxc = vrho = vsigma = wv = aow = None
        else:
            assert(all(x not in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00')))
            ao_deriv = 2
            for ao, mask, weight, coords in ni.block_loop(self.m[mol], self.scf[mol].grids, nao, ao_deriv, max_mem, non0tab):
                ngrid = weight.size
                rho = make_rho(0, ao, mask, 'MGGA')
                exc, vxc = ni.eval_xc(self.scf[mol].xc, rho, 0, 0, 1, None)[:2]
                vrho, vsigma, vlapl, vtau = vxc[:4]
                den = rho[0] * weight
                excsum[0] += (den * exc).sum()
                wv = np.empty((4,ngrid))
                wv[0]  = weight * vrho * .5
                wv[1:] = rho[1:4] * (weight * vsigma * 2)
                aow = np.einsum('npi,np->pi', ao[:4], wv)
                vmat += r_dot_product(ao[0].T,aow) #_dot_ao_ao(self.m[mol], ao[0], aow, nao, ngrid, mask)
                wv = (.5 * .5 * weight * vtau).reshape(-1,1)
                vmat += r_dot_product(ao[1].T,wv*ao[1]) #_dot_ao_ao(self.m[mol], ao[1], wv*ao[1], nao, ngrid, mask)#r_dot_product(ao[1].T,wv*ao[1])
                vmat += r_dot_product(ao[2].T,wv*ao[2])# _dot_ao_ao(self.m[mol], ao[2], wv*ao[2], nao, ngrid, mask)
                vmat += r_dot_product(ao[3].T,wv*ao[3])# _dot_ao_ao(self.m[mol], ao[3], wv*ao[3], nao, ngrid, mask)
                rho = exc = vxc = vrho = vsigma = wv = aow = None
        self.Exc[mol] = excsum[0]
        Vxc = vmat.reshape(nao,nao)
        Vxc = Vxc + Vxc.T.conj()
        return Vxc

    def _gen_rho_evaluator(self, mol, dms, hermi=1):
        natocc = []
        natorb = []
        # originally scipy
        e, c = scipy.linalg.eigh(dms)
        natocc.append(e)
        natorb.append(c)
        nao = dms.shape[0]
        ndms = len(natocc)
        def make_rho(idm, ao, non0tab, xctype):
            return eval_rhoc(mol, ao, natorb[idm], natocc[idm], non0tab, xctype)
        return make_rho, ndms, nao

    def get_jk(self, P, mol = 3):
        '''
        Args:
            P: AO density matrix
        Returns:
            J: Coulomb matrix
            K: Exchange matrix
        '''
        return self.get_j(P,mol), self.get_k(P,mol)

    def get_j(self, P, mol = 3):
        '''
        Args:
            P: AO density matrix

        Returns:
            J: Coulomb matrix (AO)
        '''
        if mol == 0 or mol == 1 or mol == 2:
            naux = self.n_aux[mol]
            nao = self.n_ao[mol]
            rho = np.einsum('ijp,ij->p', self.eri3c[mol], P)
            rho = np.linalg.solve(self.eri2c[mol], rho)
            jmat = np.einsum('p,ijp->ij', rho, self.eri3c[mol])
        else:
            print "Did not specify the Block"
        return jmat

    def get_j_c(self, P, mol = 3):
        if mol == 2:
            Pc = np.asarray(P, order='C').astype(complex)
            Jmat = np.zeros((self.n,self.n))
            #Kmat = np.asarray(Kmat,order = 'C')
            libtdscf.get_k0(\
            Pc.ctypes.data_as(ctypes.c_void_p), Jmat.ctypes.data_as(ctypes.c_void_p))
            # print Kmat
            # quit()
        return Jmat

    def get_k_c(self, P, mol = 3):
        if mol == 0:
            Pc = np.asarray(P, order='C').astype(complex)
            Kmat = np.zeros((self.nA,self.nA)).astype(complex)
            #Kmat = np.asarray(Kmat,order = 'C')
            libtdscf.get_k0(\
            Pc.ctypes.data_as(ctypes.c_void_p), Kmat.ctypes.data_as(ctypes.c_void_p))
            # print Kmat
            # quit()
        return Kmat
    def get_k(self, P, mol = 3):
        '''
        Args:
            P: BO density matrix
        Returns:
            K: Exchange matrix (BO)
        '''
        if mol == 0 or mol == 1:
            naux = int(self.n_aux[mol])
            nao = int(self.n_ao[mol])
            kpj = np.einsum('ijp,jk->ikp', self.eri3c[mol], P)
            pik = np.linalg.solve(self.eri2c[mol], kpj.reshape(-1,naux).T.conj())
            rkmat = np.einsum('pik,kjp->ij', pik.reshape(naux,nao,nao), self.eri3c[mol])
        elif mol == 2:
            # This has less overlap with High Spectra
            naux = int(self.n_aux[mol])
            nao = int(self.n_ao[mol])
            kpj = np.einsum('ijp,jk->ikp', self.eri3c[mol], P)
            pik = np.linalg.solve(self.eri2c[mol], kpj.reshape(-1,naux).T.conj()) # naux x naux, naux x n2
            rkmat = np.einsum('pik,kjp->ij', pik.reshape(naux,nao,nao)[:,:self.nA,:], self.eri3c[mol][:,:self.nA,:])
        else:
            print "Did not specify the Block"
        # print rkmat
        # quit()
        return rkmat

    def Split_RK4_Step_MMUT(self, w, v , oldrho , time, dt ,IsOn):
        Ud = np.exp(w*(-0.5j)*dt);
        U = TransMat(np.diag(Ud),v,-1)
        RhoHalfStepped = TransMat(oldrho,U,-1)
        # If any TCL propagation occurs...
        DontDo="""
        SplitLiouvillian( RhoHalfStepped, k1,tnow,IsOn);
        v2 = (dt/2.0) * k1;
        v2 += RhoHalfStepped;
        SplitLiouvillian(  v2, k2,tnow+(dt/2.0),IsOn);
        v3 = (dt/2.0) * k2;
        v3 += RhoHalfStepped;
        SplitLiouvillian(  v3, k3,tnow+(dt/2.0),IsOn);
        v4 = (dt) * k3;
        v4 += RhoHalfStepped;
        SplitLiouvillian(  v4, k4,tnow+dt,IsOn);
        newrho = RhoHalfStepped;
        newrho += dt*(1.0/6.0)*k1;
        newrho += dt*(2.0/6.0)*k2;
        newrho += dt*(2.0/6.0)*k3;
        newrho += dt*(1.0/6.0)*k4;
        newrho = U*newrho*U.t();
        """
        newrho = TransMat(RhoHalfStepped,U,-1)
#        print U.dot(U.T.conj())
#        print newrho
        return newrho

    def TDDFTstep(self,time):
        self.F = self.BO_Fockbuild(TransMat(self.rho,self.Ctilde,-1)) # BO basis
        self.F = np.conj(self.F)
        Fmo_prev = TransMat(self.F,self.Ctilde) # to MO basis

        #print self.rho

        self.eigs, rot = np.linalg.eig(Fmo_prev)
        self.rho = TransMat(self.rho, rot)

        #print self.rho

        self.rhoM12 = TransMat(self.rhoM12, rot)
        Fmo = np.diag(self.eigs).astype(complex)
        self.Ctilde = np.dot(self.Ctilde,rot)
        self.C = np.dot(self.C, rot)
        FmoPlusField, IsOn = self.field.ApplyField(Fmo,self.Ctilde,time)
        w,v = scipy.linalg.eig(FmoPlusField)
        NewRhoM12 = self.Split_RK4_Step_MMUT(w, v, self.rhoM12, time, self.params["dt"], IsOn)
        NewRho = self.Split_RK4_Step_MMUT(w, v, NewRhoM12, time,self.params["dt"]/2.0, IsOn)
        self.rho = 0.5*(NewRho+(NewRho.T.conj()));

        #print self.rho

        #quit()

        self.rhoM12 = 0.5*(NewRhoM12+(NewRhoM12.T.conj()))

    def dipole(self, AA = False):
        if AA == False:
            return self.field.Expectation(self.rho, self.C, AA)
        else:
            return self.field.Expectation(self.rho, self.C, AA, self.nA,self.U)
    def loginstant(self,iter):
        """
        time is logged in atomic units.
        """
        np.set_printoptions(precision = 7)
        tore = str(self.t)+" "+str(self.dipole().real).rstrip(']').lstrip('[')+ " " +str(self.energy(TransMat(self.rho,self.Ctilde,-1),False))+" "+str(np.trace(self.rho))
        #tore = str(self.t)+" "+str(np.sin(0.5*np.ones(3)*self.t)).rstrip(']').lstrip('[]')+ " " +str(self.energyc(TransMat(self.rho,self.C,-1),False)+self.Enuc)
        #print tore
        if iter%self.params["StatusEvery"] ==0 or iter == self.params["MaxIter"]-1:
            print 't:', self.t*FsPerAu, " (Fs)  Energy:",self.energy(TransMat(self.rho,self.Ctilde,-1)), " Tr ",(np.trace(self.rho))
            print('Dipole moment(X, Y, Z, au): %8.5f, %8.5f, %8.5f' %(self.dipole().real[0],self.dipole().real[1],self.dipole().real[2]))
        return tore

    def AAlog(self,iter):
        """
        time is logged in atomic units.
        """
        np.set_printoptions(precision = 7)
        tore = str(self.t)+" "+str(self.dipole(True).real).rstrip(']').lstrip('[')
        return tore


    def step(self,time):
        """
        Performs a step
        Updates t, rho, and possibly other things.
        """
        return self.TDDFTstep(time)



    def prop(self,output):
        """
        The main tdscf propagation loop.
        """
        iter = 0
        self.t = 0
        f = open(output,'a')
        start = time.time()
        aa = open(output + '.aa','a')
        print "Energy Gap (eV)",abs(self.eigs[self.n_occ]-self.eigs[self.n_occ-1])*27.2114
        print "\n\nPropagation Begins"
        while (iter<self.params["MaxIter"]):
            self.step(self.t)
            f.write(self.loginstant(iter)+"\n")
            if (self.AA == True):
                aa.write(self.AAlog(iter) + "\n")
            if iter%self.params["StatusEvery"] ==0 or iter == self.params["MaxIter"]-1:
                end = time.time()
                if iter == 0:
                    print "0"
                else:
                    print (end - start)/(self.t * FsPerAu), "sec/fs"
            iter = iter + 1
            self.t = self.t + self.params["dt"]
        f.close()
        aa.close()





def BOprojector(mol1,mol2):
    S = scf.hf.get_ovlp(mol2)
    nA = gto.nao_nr(mol1)
    nAB = gto.nao_nr(mol2)
    nB = nAB - nA
    SAA = S[0:nA, 0:nA]
    SBB = S[nA:nAB, nA:nAB]
    SAB = S[0:nA,nA:nAB]
    U = np.identity(nAB)
    PAB = np.dot(np.linalg.inv(SAA), SAB)
    U[0:nA,nA:nAB] = -PAB
    return U


def dmguess1 (mol,U):
    """
    Solve for Initial Density using H
    """
    m = scf.RHF(mol).run()
    H = m.get_hcore()
    S = m.get_ovlp()
    occ = m.mo_occ
    rho = np.diag(occ)
    # Transf. MO to AO (rho => P) verify that tr(PS) = Sum(mo_occ) Guranteed by unitary property
    # Note: A-block subtrace of the above P (P_aa) must equal the A-block subtrace of this guess
    #eigval, eigvec = np.linalg.eig(H)
    eigval, eigvec = scf.hf.eig(H,S)
    # TransMat(eigvec): AO to MO
    # TransMat(eigvec) -1: MO to AO
    # rho(MO) -> P
    P = TransMat(rho,eigvec,-1)
    # TM(U): AO to BO
    Ptilde = TransMat(P,U,-1)
    #mH = TransMat(H,U)
    #eigval, eigvec = np.linalg.eig(mH) # Htilde IS NOT ORTHOGONAL ie: you cannot make a guess by putting one's on it's diagonal eigenvectors.
    # instead solve (29) where Ftilde = Hcore, and use the resulting Ctilde to construct Dtilde
    return Ptilde


def dmguess(rks1,U):
    """
    Somewhat Same as dmguess1; It results in better Number of electron
    """
    #m = scf.RHF(mol).run()
    H = rks1.get_hcore()
    Htilde = TransMat(H,U)
    S = rks1.get_ovlp()
    Stilde = TransMat(S,U)
    eigval, eigvec = scf.hf.eig(Htilde,Stilde)
    occ = rks1.mo_occ
    mo = np.diag(occ)
    # TransMat(eigvec) -1: MO to BO
    Ptilde = TransMat(mo,eigvec,-1)
    return Ptilde





def NeCheck(D,S,U,nA,inv = 1):
    """
    Checking if Ne in AO basis equals Ne in BO basis
    """
    mD = TransMat(D,U,inv)
    mS = TransMat(S,U)
    n = D.shape[0]
    Ne0 = TrDot(D,S)
    Ne1 = TrDot(mD,mS)
    Ne2 = TrDot(mD[0:nA,0:nA],S[0:nA,0:nA]) + TrDot(mD[nA:n,nA:n],mS[nA:n,nA:n])
    if abs(Ne1-Ne2) < 10**-4:
        print "Tr(mD*mS) = Tr(mDAA*SAA) + Tr(mDBB*mSBB)"
    else:
        print "Tr(mD*mS) != Tr(mDAA*SAA) + Tr(mDBB*mSBB)"
        print "Off by", Ne1-Ne2
    if abs(Ne0 - Ne1) > 10**-4:
        print "Ne off by ", Ne0 - Ne1
    else:
        print "Ne", Ne1

def FockMat(mol):
    m = scf.RHF(mol)
    m.scf()
    dm = m.make_rdm1()
    H = scf.hf.get_hcore(mol)
    S = scf.hf.get_ovlp(mol)
    V = scf.hf.get_veff(mol,dm)
    F = scf.hf.get_fock(m,H,S,V,dm)
    return F

def Ghf(mol,dm):
    eri = scf._vhf.int2e_sph(mol._atm, mol._bas, mol._env)
    eri1 = ao2mo.restore(1, eri, mol.nao_nr())
    n = eri1.shape[0]
    G = np.zeros((n,n))
    for i in range(n):
        for ip in range(n):
            for j in range(n):
                for jp in range(n):
                    G[i,j] += 2*eri1[j,i,ip,jp]*dm[ip,jp]
                    G[i,j] -= eri1[i,jp,ip,j]*dm[ip,jp]
    return G

def EEX1(mol,dm,dmaa):
    eri = scf._vhf.int2e_sph(mol._atm, mol._bas, mol._env)
    eri1 = ao2mo.restore(1, eri, mol.nao_nr())
    na = dmaa.shape[0]
    n = eri1.shape[0]
    G = 0
    for i in range(0,na):
        for ip in range(0,na):
            for j in range(0,na):
                for jp in range(0,na):
                    G -= -0.25 * eri1[i,ip,j,jp]*dmaa[i,j]*dmaa[ip,jp]
    for i in range(0,na):
        for j in range(0,na):
            for ip in range(na,n):
                for jp in range(na,n):
                    G -= -0.25 * eri1[i,ip,j,jp]*dmaa[i,j]*dm[ip,jp]
    return G
def z2cart(a):
    acart = gto.mole.from_zmatrix(a)
    b = ''
    for x in acart:
        coord = str(x[1])
        coord = coord.rstrip(']')
        coord = coord.lstrip('[')
        b += x[0] + " " + coord + "\n"
    return b
