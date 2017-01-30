from tdscf import *
import pyscf
import pyscf.dft
np.set_printoptions(linewidth=220, suppress = True,precision = 7)


def TestTDSCF():
    """
    Tests Basic Propagation Functionality.
    """
    geom = """     H        0.00000         0.00000         0.75000
                   H        0.00000         0.00000         0.00000""".split("\n")
    pyscfatomstring=""
    for line in geom:
        s = line.split()
        pyscfatomstring=pyscfatomstring+s[0]+" "+str(s[1])+" "+str(s[2])+" "+str(s[3])+";"
    # Here finish an SCF calculation and pass it to TDSCF.
    mol = gto.Mole()
    mol.atom = pyscfatomstring
    mol.basis = 'sto-6g'
    mol.build()
    the_scf = pyscf.dft.RKS(mol)
    the_scf.xc='PBE'
    print "Inital SCF finished. E=", the_scf.kernel()
    aprop = tdscf(the_scf)
    return

TestTDSCF()
