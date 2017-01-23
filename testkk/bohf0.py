from pyscf.tdscf.bo import *
import numpy as np
np.set_printoptions(linewidth=220, suppress = True,precision = 7)

print "Using the same basis"

f = open('hf.out','a')
#Input Section
n = 10
inc = 0.1
for i in range(-n,2*n):
	z1 = '''
	O
	H  1  0.9870
	H  1  0.9739  2  109.01
	O  2  '''
	z2 =  str(1.7958 + inc*i)
	z3='''  1  172.27  3  156.75
	H  4  0.9745  2  113.07  1  83.29
	H  4  0.9747  2  109.58  1  320.76
               '''
	z = z1 + z2 + z3


	a = z2cart(z)
	print "\n\nH bond length(AA):",z2
	bas = ['6-31g','6-31g']
	# End of Input



	bo1 = BORHF(a,bas,3)
	f.write(z2+" "+str(bo1.EBO)+ " " + str(bo1.hf3.e_tot) + "\n")

f.close()
