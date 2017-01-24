from TCL.bo import *
import numpy as np
np.set_printoptions(linewidth=220, suppress = True,precision = 7)

"""
mol1:high, AA
mol2:low, AA
mol3:low, whole
"""

print "Test 1: H2 with 2 H2O"
#Input Seciton
a = '''
               H        0.00000         0.00000         0.75000
               H        0.00000         0.00000         0.00000
               O       -5.1777000982      1.6901035747      0.0002889160
               H       -4.2077000982      1.6901035747      0.0002889160
               H       -5.5010299099      2.0328140482     -0.8475951351
               O       -4.2723968664     -0.5739974456      0.0002889160
               H       -3.3023968664     -0.5739974456      0.0002889160
               H       -4.5957266781      0.0467098976     -0.6713361483
               '''

bas = ['6-31g','sto-6g']
xc = ['b3lyp','lda']
# End of Input



bo1 = BORKS(a,bas,2,xc)
print "\n\n\n\nFinished Setup"
print "Total Energy with lda:", bo1.rks3.e_tot
print "\n\nEnergy in Original Basis"
dm1= bo1.rks3.make_rdm1()
Ecore = TrDot(bo1.H, dm1)
E2 = TrDot(dft.rks.get_veff(bo1.rks3, None,dm1),dm1)
E= Ecore + E2
print "Ecore                ",Ecore
print "2e E                 ",E2
print "Electronic Energy    ",E
print "Total E              ",E + bo1.Enuc
print "\n\nEnergy in BO basis"
E = TrDot(bo1.Ftilde, bo1.Ptilde)
print "Electronic Energy:   ",E
print "Total Energy         ",E + bo1.Enuc

print "\nEnuc", bo1.Enuc

Ne = TrDot(bo1.Ptilde,bo1.Stilde)
print "Ne",Ne
#FockBuild
