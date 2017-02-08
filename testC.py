from tdscf import *
import pyscf, sys, re
import pyscf.dft
np.set_printoptions(linewidth=220, suppress = True,precision = 7)

def TestTDSCF():
    """
    Tests Basic Propagation Functionality.
    """
    prm = '''
	Model  TDCI
	Method CIS

	dt      0.02
	MaxIter 100

	ExDir   1.0
	EyDir   1.0
	EzDir   1.0
	FieldAmplitude  0.01
	FieldFreq       0.9202
	ApplyImpulse    1
	ApplyCw         0

	StatusEvery     10
	'''

    geom = """     H        0.00000         0.00000         0.75000
                   H        0.00000         0.00000         0.00000""".split("\n")
    pyscfatomstring=""
    for line in geom:
        s = line.split()
        pyscfatomstring=pyscfatomstring+s[0]+" "+str(s[1])+" "+str(s[2])+" "+str(s[3])+";"
    # Here finish an SCF calculation and pass it to TDSCF.

    output = re.sub("py","dat",sys.argv[0])
    mol = gto.Mole()
    mol.atom = pyscfatomstring
    mol.basis = 'sto-6g'
    mol.build()
    the_scf = pyscf.dft.RKS(mol)
    the_scf.xc='PBE'
    print "Inital SCF finished. E=", the_scf.kernel()
    aprop = tdscfC(the_scf,prm,output)
    return

TestTDSCF()
