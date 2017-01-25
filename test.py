from tdscf import *
import pyscf
import pyscf.dft
np.set_printoptions(linewidth=220, suppress = True,precision = 7)


def TestTDSCF():
    """
    Tests Basic Propagation Functionality.
    """
    geom = """     H        0.00000         0.00000         0.75000
                   H        0.00000         0.00000         0.00000
                   O       -5.1777000982      1.6901035747      0.0002889160
                   H       -4.2077000982      1.6901035747      0.0002889160
                   H       -5.5010299099      2.0328140482     -0.8475951351
                   O       -4.2723968664     -0.5739974456      0.0002889160
                   H       -3.3023968664     -0.5739974456      0.0002889160
                   H       -4.5957266781      0.0467098976     -0.6713361483""".split("\n")
    pyscfatomstring=""
    for line in geom:
        s = line.split()
        pyscfatomstring=pyscfatomstring+s[0]+" "+str(s[1])+" "+str(s[2])+" "+str(s[3])+";"
    # Here finish an SCF calculation and pass it to TDSCF.
    mol = gto.Mole()
    mol.atom = pyscfatomstring
    mol.basis = '6-31G'
    mol.build()
    the_scf = pyscf.dft.RKS(mol)
    the_scf.xc='PBE'
    print "Inital SCF finished. E=", the_scf.kernel()
    aprop = tdscf(the_scf)
    return

TestTDSCF()
