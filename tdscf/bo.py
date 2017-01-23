import numpy as np
import scipy
import scipy.linalg
from pyscf import gto, dft, scf, ao2mo

class BORHF():
    def __init__(self, atom, basis, n):
        """
        Make a BlockOrthogonalizedRHF
        Args: mol1_, mol2_ fragments to make a block orthogonal SCF from.
        """
        #mol objects
        self.m1, self.m2, self.m3 = self.MixedBasisMol(atom,basis,n)
        # rks objects
        self.hf1, self.hf2, self.hf3 = self.MixedTheory()
        # Different Matrix
        # U: |AO><BO|
        self.U = BOprojector(self.m1,self.m3)
        # O: |BO><MO|
        self.O = None
        self.Vtilde = None
        self.Enuc = self.hf3.e_tot - scf.hf.energy_elec(self.hf3,self.hf3.make_rdm1())[0]
        self.H = self.hf3.get_hcore()
        self.Htilde = TransMat(self.H,self.U)
        self.S = self.hf3.get_ovlp()
        self.Stilde = TransMat(self.S,self.U)
        # Constants
        self.nA = self.m1.nao_nr()
        self.n = self.m3.nao_nr()
        # output
        self.EBO = None


        self.Ptilde = self.GetGuess()
        self.Ftilde = self.FockGuess()

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

    def MixedTheory(self):
        # Generate Mixed HF objects
        m1 = scf.hf.RHF(self.m1)
        m1.kernel()
        m2 = scf.hf.RHF(self.m2)
        m2.kernel()
        m3 = scf.hf.RHF(self.m3)
        m3.kernel()

        return m1,m2,m3

    def AOHF(self):
        dm3 = self.hf3.make_rdm1()
        Ne = TrDot(dm3,self.S)
        J,K = scf.hf.get_jk(self.m3,dm3)
        print "AO based Calculation"
        print "--------------------------------------"
        print "Ne(AO):                  ",Ne
        print "1e Energy:               ",TrDot(dm3, self.H)
        print "Coulomb Energy           ", 0.5 * TrDot(dm3,J)
        print "Exchange Energy          ", -0.25 * TrDot(dm3,K)
        print "Nuclear Repulsion Energy:", self.Enuc
        Eref = TrDot(dm3, self.H)+0.5 * TrDot(dm3,J)+ -0.25 * TrDot(dm3,K) + self.Enuc
        print "Total Energy in AO Basis:",Eref
        print "--------------------------------------"

        return

    def FockGuess(self):
        print "======================================"
        print "Roothan's Equation"
        dm3 = self.hf3.make_rdm1()
        Ptilde = TransMat(dm3,self.U)
        Stilde = TransMat(self.S,self.U)
        Ftilde = self.BOFockBuild(Ptilde)
        noc = int(sum(self.hf3.mo_occ)/2)
        n = self.n
        err = 100
        it = 0
        Pold = Ptilde
        self.AOHF()
        Ne = TrDot(dm3,self.S)
        print "--------------------------------------"
        Eb = self.BOEnergy(Ptilde) + self.Enuc
        print "E before Solution(BO):   ",Eb, "\n\n"
        print "--------------------------------------"
        while (err > 10**-10):
            Fold = Ftilde
            etilde, Ctilde = scipy.linalg.eigh(Ftilde, Stilde)
            # C: |BO><MO|
            # Ptilde/2 = C*C.t() for Occ orbitals
            Pold = Ptilde
            Ptilde = 2 * np.dot(Ctilde[:,:noc],np.transpose(Ctilde[:,:noc]))
            Ptilde = 0.6*Pold + 0.4*Ptilde
            Ptilde = Ne * Ptilde / (TrDot(Ptilde, Stilde))
            Ftilde = self.BOFockBuild(Ptilde)
            err = abs(sum(sum(Ftilde-Fold)))
            if (it%10 ==0):
                print "Iteration:", it,"; Error =",err
            it += 1
        print "--------------------------------------"
        print "Final Iteration:",it
        EBO = self.BOEnergy(Ptilde) + self.Enuc
        self.EBO = EBO
        print "E after Solution(BO):",EBO
        print "--------------------------------------"
        nA = self.nA
        n = self.n
        print "--------------------------------------"
        print "Ne:                  ",TrDot(Ptilde, Stilde), "\n\n"
        print "Ne of AA", TrDot(Ptilde[0:nA,0:nA], self.S[0:nA,0:nA])
        print "Ne of BB", TrDot(Ptilde[nA:n,nA:n], self.Stilde[nA:n,nA:n])
        print "--------------------------------------"


        self.Ptilde = Ptilde
        return Ftilde

    def BOFockBuild(self,Ptilde = None):
        nA = self.nA
        n = self.n

        Htilde = TransMat(self.H,self.U)

        P = TransMat(Ptilde,self.U,-1)
        V = scf.hf.get_veff(self.m3,P)
        Vtilde = TransMat(V,self.U)

        self.Vtilde = Vtilde
        Ftilde = Htilde + Vtilde
        return Ftilde

    def AAEnergy(self,Ptilde):
        nA = self.nA
        n = self.n

        newP = np.zeros((n,n))
        newP[0:nA,0:nA] = Ptilde[0:nA,0:nA]

        Ecore = TrDot(newP, self.Htilde)
        dft.rks.get_veff(self.rks3,None,newP)
        dft.rks.get_veff(self.rks1,None,newP[0:nA,0:nA])
        Ecoul = self.rks3._ecoul
        Exc = self.rks1._exc

        E = Ecore + Ecoul + Exc
        return E

    def BBEnergy(self,Ptilde):
        nA = self.nA
        n = self.n

        newP = np.zeros((n,n))
        newP[nA:n,nA:n] = Ptilde[nA:n,nA:n]

        Ecore = TrDot(newP,self.Htilde)
        dft.rks.get_veff(self.rks3,None,newP)
        Ecoul = self.rks3._ecoul
        Exc = self.rks3._exc

        E = Ecore + Ecoul + Exc
        return E

    def ABEnergy(self,Ptilde):
        nA = self.nA
        n = self.n

        newP = np.zeros((n,n))
        newP[0:nA,nA:n] = Ptilde[0:nA,nA:n]
        newP[nA:n,0:nA] = Ptilde[nA:n,0:nA]

        Ecore = TrDot(newP,self.Htilde)
        dft.rks.get_veff(self.rks3,None,newP)
        Ecoul = self.rks3._ecoul
        Exc = self.rks3._exc

        E = Ecore + Ecoul + Exc
        return E



    def BOEnergy(self,Ptilde):
        nA = self.nA
        P = TransMat(Ptilde,self.U,-1)


        Ecore = TrDot(Ptilde, self.Htilde)

        J,K = scf.hf.get_jk(self.m3,P)
        Jtilde = TransMat(J,self.U)
        Ktilde = TransMat(K,self.U)
        EJ = 0.5*TrDot(Ptilde,Jtilde)
        EK = -0.25*TrDot(Ptilde,Ktilde)

        print "Ecore    ",Ecore
        print "EJ       ",EJ
        print "EK       ",EK

        E = Ecore + EJ + EK
        return E

    def GetGuess(self):
        """
        Assign self.Ptilde to a good initial guess.
        returns: a valid P-tilde.
        """
        Htilde = TransMat(self.H,self.U)
        eigval, self.O = scipy.linalg.eigh(Htilde,self.Stilde)
        occ = self.hf3.mo_occ
        mo = np.diag(occ)
        # TransMat(O) -1: MO to BO
        Ptilde = TransMat(mo,self.O,-1)
        return Ptilde


class BORKS():
    def __init__(self, atom, basis, n, xc, BB = 0):
        """
        Make a BlockOrthogonalizedRHF
        Args: mol1_, mol2_ fragments to make a block orthogonal SCF from.
        """

        # bb ojbect(optional)
        self.BB = BB
        self.mb = None
        self.rksb = None
        #mol objects
        self.m1, self.m2, self.m3 = self.MixedBasisMol(atom,basis,n)
        # rks objects
        self.rks1, self.rks2, self.rks3 = self.MixedTheory(xc)
        # TO BE DELETED later
        self.AOExc = None
        self.BOExc = None
        # Different Matrix
        # U: |AO><BO|
        self.U = BOprojector(self.m1,self.m3)
        # O: |BO><MO|
        self.O = None
        self.Vtilde = None
        self.Enuc = self.rks3.e_tot - dft.rks.energy_elec(self.rks3,self.rks3.make_rdm1())[0]
        self.H = self.rks3.get_hcore()
        self.Htilde = TransMat(self.H,self.U)
        self.S = self.rks3.get_ovlp()
        self.Stilde = TransMat(self.S,self.U)
        # Constants
        self.nA = self.m1.nao_nr()
        self.n = self.m3.nao_nr()
        # output
        self.EBO = None


        self.Ptilde = self.GetGuess()
        self.Ftilde = self.FockGuess()

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

        #BB molecule
        if self.BB != 0:
            molbb = gto.Mole()
            molbb.atom = atm2
            molbb.basis = bas[1]
            molbb.build()
            self.mb = molbb

        return mol1,mol2,mol3

    def MixedTheory(self,xc):
        # Generate Mixed XC dft objects
        m1 = dft.rks.RKS(self.m1)
        m1.xc = xc[0]
        print "High Theory AA"
        m1.kernel()
        m2 = dft.rks.RKS(self.m2)
        m2.xc = xc[1]
        print "Low Theory AA"
        m2.kernel()
        m3 = dft.rks.RKS(self.m3)
        m3.xc = xc[1]
        print "Low Theory"
        m3.kernel()
        # optional BB
        if self.BB != 0:
            print "Low Theory BB(Optional): On"
            rksb = dft.rks.RKS(self.mb)
            rksb.xc = xc[1]
            rksb.kernel()
            self.rksb = rksb
        return m1,m2,m3

    def FockGuess(self):
        print "\n\n======================================"
        print "Roothan's Equation"
        print "======================================"
        dm3 = self.rks3.make_rdm1()
        nA = self.nA
        Ptilde = TransMat(dm3,self.U)
        Stilde = TransMat(self.S,self.U)
        Ftilde = self.BOFockBuild(Ptilde)
        noc = int(sum(self.rks3.mo_occ)/2)
        n = self.n
        err = 100
        it = 0
        Pold = Ptilde
        Ne = TrDot(dm3,self.S)
        Veff = dft.rks.get_veff(self.rks3,None,dm3)
        J = self.rks3.get_j(self.m3,dm3,1)
        Vxc = Veff - J
        print "Ne(AO):                  ",Ne
        print "1e Energy:               ",TrDot(dm3, self.H)
        print "Ecoul:                   ",self.rks3._ecoul
        print "Exc:                     ", self.rks3._exc
        print "Nuclear Repulsion Energy:", self.Enuc
        Eref = TrDot(dm3, self.H)+self.rks3._ecoul+ self.rks3._exc + self.Enuc
        print "Total Energy in AO Basis:",Eref
        print "--------------------------------------"
        while (err > 10**-10):
            Fold = Ftilde
            etilde, Ctilde = scipy.linalg.eigh(Ftilde, Stilde)
            # C: |BO><MO|
            # Ptilde/2 = C*C.t() for Occ orbitals
            Pold = Ptilde
            Ptilde = 2 * np.dot(Ctilde[:,:noc],np.transpose(Ctilde[:,:noc]))
            Ptilde = 0.6*Pold + 0.4*Ptilde
            Ptilde = Ne * Ptilde / (TrDot(Ptilde, Stilde))
            Ftilde = self.BOFockBuild(Ptilde)
            err = abs(sum(sum(Ftilde-Fold)))
            if (it%10 ==0):
                print "Iteration:", it,"; Error =",err
            it += 1
        print "--------------------------------------"
        print "Final Iteration:",it

        EBO = self.BOEnergy(Ptilde) + self.Enuc
        self.EBO = EBO

        print "E after Solution(BO):",EBO
        print "SCF energy(RKS):     ", Eref
        print "--------------------------------------"
        print "Ne:                  ",TrDot(Ptilde, self.Stilde), "\n\n"
        print "Ne of AA", TrDot(Ptilde[0:nA,0:nA], self.S[0:nA,0:nA])
        print "Ne of BB", TrDot(Ptilde[nA:n,nA:n], self.Stilde[nA:n,nA:n])
        print "PARTITION INFORMATION"
        print "--------------------------------------"
        if self.BB != 0:
            HAA = self.rks1.get_hcore()
            p1 = self.rks1.make_rdm1()
            EnucAA = self.rks1.e_tot - dft.rks.energy_elec(self.rks1,p1)[0]

            pb = self.rksb.make_rdm1()
            EnucBB = self.rksb.e_tot - dft.rks.energy_elec(self.rksb,pb)[0]
            print "Using AO Density"
            print "1e Energy                ",TrDot(p1, HAA)
            print "Ecoul                    ",self.rks1._ecoul
            print "Exc                      ", self.rks1._exc
            print "Total Energy in AO Basis ",TrDot(p1, HAA)+self.rks1._ecoul+ self.rks1._exc + EnucAA
            print "--------------------------------------"
            # The Core Energy of AA and BB are separated Correctly
            # There are some difference in nuclear energy (understandable)
            # 2e is lacking too
            print "BB Energy(BO):       ", self.BBEnergy(Ptilde) + EnucBB
            print "BB Nuc E:            ", EnucBB
            EA = self.AAEnergy(Ptilde) + EnucAA
            EB = self.BBEnergy(Ptilde) + EnucBB
            EAB = EA + EB
            print "AA Energy + BB Energy", EAB
            print "Difference in Total: ", EBO-EAB
            print "Difference in Enuc   ", self.Enuc - EnucAA - EnucBB
            print "AA(After)            ", EA + 0.5*(EBO-EAB)
            print "BB(After)            ", EB + 0.5*(EBO-EAB)
            print "AB Energy            ", self.ABEnergy(Ptilde)
        print "--------------------------------------\n\n\n\n\n"


        self.Ptilde = Ptilde
        return Ftilde

    def BOFockBuild(self,Ptilde = None):
        nA = self.nA
        n = self.n

        Htilde = TransMat(self.H,self.U)
        # High AA
        VhtildeAA = dft.rks.get_veff(self.rks1,None,Ptilde[0:nA,0:nA])
        VB = np.zeros((n,n))
        VB[0:nA,0:nA] = VhtildeAA[0:nA,0:nA]
        # Low AA
        VltildeAA = dft.rks.get_veff(self.rks2,None,Ptilde[0:nA,0:nA])
        VA = np.zeros((n,n))
        VA[0:nA,0:nA] = VltildeAA[0:nA,0:nA]
        # Low Whole
        P = TransMat(Ptilde,self.U,-1)
        Vlow = dft.rks.get_veff(self.rks3,None,P)
        Vltilde = TransMat(Vlow,self.U)

        self.Vtilde = Vltilde - VA + VB
        Ftilde = Htilde + Vltilde - VA + VB
        return Ftilde

    def AAEnergy(self,Ptilde):
        nA = self.nA
        n = self.n

        newP = np.zeros((n,n))
        newP[0:nA,0:nA] = Ptilde[0:nA,0:nA]

        Ecore = TrDot(newP, self.Htilde)
        dft.rks.get_veff(self.rks3,None,newP)
        dft.rks.get_veff(self.rks1,None,newP[0:nA,0:nA])
        Ecoul = self.rks3._ecoul
        Exc = self.rks1._exc

        E = Ecore + Ecoul + Exc
        return E

    def BBEnergy(self,Ptilde):
        nA = self.nA
        n = self.n

        newP = np.zeros((n,n))
        newP[nA:n,nA:n] = Ptilde[nA:n,nA:n]

        Ecore = TrDot(newP,self.Htilde)
        dft.rks.get_veff(self.rks3,None,newP)
        Ecoul = self.rks3._ecoul
        Exc = self.rks3._exc

        E = Ecore + Ecoul + Exc
        return E

    def ABEnergy(self,Ptilde):
        nA = self.nA
        n = self.n

        newP = np.zeros((n,n))
        newP[0:nA,nA:n] = Ptilde[0:nA,nA:n]
        newP[nA:n,0:nA] = Ptilde[nA:n,0:nA]

        Ecore = TrDot(newP,self.Htilde)
        dft.rks.get_veff(self.rks3,None,newP)
        Ecoul = self.rks3._ecoul
        Exc = self.rks3._exc

        E = Ecore + Ecoul + Exc
        return E



    def BOEnergy(self,Ptilde):
        nA = self.nA
        P = TransMat(Ptilde,self.U,-1)


        Ecore = TrDot(Ptilde, self.Htilde)

        J,K = scf.hf.get_jk(self.m3,P)


        Ktilde = TransMat(K,self.U)
        Ecoul = 0.5*TrDot(P,J)

        nn, Elow, Vx = self.rks3._numint.nr_rks(self.m3, self.rks3.grids, self.rks3.xc, P, 0)
        hyb = self.rks3._numint.hybrid_coeff(self.rks3.xc, spin=(self.m3.spin>0)+1)
        Elow -= 0.5 * 0.5 * hyb * TrDot(P,K)

        Ptilde1 = np.zeros((self.n,self.n))
        Ptilde1[0:nA,0:nA]=Ptilde[0:nA,0:nA]
        P1 = TransMat(Ptilde1,self.U,-1)
        P1[0:nA,0:nA]=P[0:nA,0:nA]

        nn, EHAA, Vx = self.rks3._numint.nr_rks(self.m3, self.rks3.grids, self.rks1.xc, P1, 0)
        hyb = self.rks1._numint.hybrid_coeff(self.rks1.xc, spin=(self.m1.spin>0)+1)
        EHAA -= 0.5 * 0.5 * hyb * TrDot(Ptilde1,Ktilde)

        nn, ELAA, Vx = self.rks3._numint.nr_rks(self.m3, self.rks3.grids, self.rks2.xc, P1, 0)
        hyb = self.rks2._numint.hybrid_coeff(self.rks2.xc, spin=(self.m2.spin>0)+1)
        ELAA -= 0.5 * 0.5 * hyb * TrDot(Ptilde1,Ktilde)



        print "Ecore:       ",Ecore
        print "Ecoul:       ",Ecoul
        print "Exc:         ",Elow
        print "ExcHigh      ",EHAA
        print "ExcLow       ",ELAA
        E = Ecore + Ecoul + Elow - ELAA + EHAA
        return E

    def GetGuess(self):
        """
        Assign self.Ptilde to a good initial guess.
        returns: a valid P-tilde.
        """
        Htilde = TransMat(self.H,self.U)
        eigval, self.O = scipy.linalg.eigh(Htilde,self.Stilde)
        occ = self.rks3.mo_occ
        mo = np.diag(occ)
        # TransMat(O) -1: MO to BO
        Ptilde = TransMat(mo,self.O,-1)
        return Ptilde




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

def TransMat(M,U,inv = 1):
    if inv == 1:
        # U.t() * M * U
        Mtilde = np.dot(np.dot(np.transpose(U),M),U)
    elif inv == -1:
        # U * M * U.t()
        Mtilde = np.dot(np.dot(U,M),np.transpose(U))
    return Mtilde

def TrDot(A,B):
    C = np.trace(np.dot(A,B))
    return C


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
