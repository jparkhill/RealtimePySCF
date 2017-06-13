import numpy as np
from tdscf.tdscf import tdscf
import sys, re
import pyscf
import pyscf.dft
from  pyscf import gto
np.set_printoptions(linewidth=220, suppress = True,precision = 7)

def TestTDHF():
    """
    Tests Basic Propagation Functionality. TDHF
    """
    prm = '''
    Model	TDHF
    Method	MMUT
    dt	0.02
    MaxIter	5000
    ExDir	1.0
    EyDir	1.0
    EzDir	1.0
    FieldAmplitude	0.01
    FieldFreq	0.9202
    ApplyImpulse	1
    ApplyCw		0
    StatusEvery	2000
    '''
    geom = """
    H 0. 0. 0.
    H 0. 0. 0.9
    H 2.0 0.  0
    H 2.0 0.9 0
    """
    output = re.sub("py","dat",sys.argv[0])
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = 'sto-3g'
    mol.build()
    the_scf = pyscf.dft.RKS(mol)
    the_scf.xc='HF'
    print "Inital SCF finished. E=", the_scf.kernel()
    aprop = tdscf(the_scf,prm,output)
    return

def TestTDDFT():
    """
    Tests Basic Propagation Functionality. TDDFT
    """
    prm = '''
    Model	TDDFT
    Method	MMUT
    dt	0.02
    MaxIter	5000
    ExDir	1.0
    EyDir	1.0
    EzDir	1.0
    FieldAmplitude	0.01
    FieldFreq	0.9202
    ApplyImpulse	1
    ApplyCw		0
    StatusEvery	2000
    '''
    geom = """
    H 0. 0. 0.
    H 0. 0. 0.9
    H 2.0 0.  0
    H 2.0 0.9 0
    """
    output = re.sub("py","dat",sys.argv[0])
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = 'sto-3g'
    mol.build()
    the_scf = pyscf.dft.RKS(mol)
    the_scf.xc='PBE'
    print "Inital SCF finished. E=", the_scf.kernel()
    aprop = tdscf(the_scf,prm,output)
    return

def TestProfileTDDFT():
    """
    Tests Basic Propagation Functionality. TDDFT
    """
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    prm = '''
    Model	TDDFT
    Method	MMUT
    dt	0.02
    MaxIter	5000
    ExDir	1.0
    EyDir	1.0
    EzDir	1.0
    FieldAmplitude	0.01
    FieldFreq	0.9202
    ApplyImpulse	1
    ApplyCw		0
    StatusEvery	2000
    '''
    geom = """
    H 0. 0. 0.
    H 0. 0. 0.9
    H 2.0 0.  0
    H 2.0 0.9 0
    """
    output = re.sub("py","dat",sys.argv[0])
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = 'sto-3g'
    mol.build()
    the_scf = pyscf.dft.RKS(mol)
    the_scf.xc='PBE'
    print "Inital SCF finished. E=", the_scf.kernel()
    pr.enable()
    aprop = tdscf(the_scf,prm,output)
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    return


#TestProfileTDDFT()
#TestTDDFT()
TestTDHF()
