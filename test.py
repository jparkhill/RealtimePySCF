from tdscf import *

def TestTDSCF():
    """
    Tests Basic Propagation Functionality. 
    """
    geom = '''
                   H        0.00000         0.00000         0.75000
                   H        0.00000         0.00000         0.00000
                   O       -5.1777000982      1.6901035747      0.0002889160
                   H       -4.2077000982      1.6901035747      0.0002889160
                   H       -5.5010299099      2.0328140482     -0.8475951351
                   O       -4.2723968664     -0.5739974456      0.0002889160
                   H       -3.3023968664     -0.5739974456      0.0002889160
                   H       -4.5957266781      0.0467098976     -0.6713361483
                   '''
    # Here finish an SCF calculation and pass it to TDSCF.
    the_scf = pyscf.scf.RKS(mol)
    aprop = tdscf(the_scf)
    return

TestTDSCF()
