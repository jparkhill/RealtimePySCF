import numpy as np
from tdscf import tdscf
import pyscf
import pyscf.dft
from  pyscf import gto 
np.set_printoptions(linewidth=220, suppress = True,precision = 7)

def TestTDSCF():
    """
    Tests Basic Propagation Functionality.
    """
    geom = """H 0. 0. 0.
H 0. 0. 0.9
H 2.0 0.  0
H 2.0 0.9 0""".split("\n")
    pyscfatomstring=""
    for line in geom:
        s = line.split()
        pyscfatomstring=pyscfatomstring+s[0]+" "+str(s[1])+" "+str(s[2])+" "+str(s[3])+";"
    # Here finish an SCF calculation and pass it to TDSCF.
    mol = gto.Mole()
    mol.atom = pyscfatomstring
    mol.basis = 'sto-3g'
    mol.build()
    the_scf = pyscf.dft.RKS(mol)
    the_scf.xc='HF'
    print "Inital SCF finished. E=", the_scf.kernel()
    aprop = tdscf(the_scf)
    return

TestTDSCF()
