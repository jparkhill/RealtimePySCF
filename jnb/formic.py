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

    geom = """    
	O          1.45700       -0.06080        0.51300
	C          1.92776        0.24064       -0.56750
	O          3.25265        0.29382       -0.77255
	H          1.36836        0.49797       -1.47872
	H          3.68026        0.85182        0.22298
           """

    output = re.sub("py","dat",sys.argv[0])
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = 'sto-3g'
    mol.build()
    the_scf = pyscf.dft.RKS(mol)
    the_scf.xc='PBE'
    print "Inital SCF finished. E=", the_scf.kernel()
    aprop = tdscfC(the_scf,prm,output)
    return

TestTDSCF()
