from pyscf.tdscf.bo import *
import numpy as np
np.set_printoptions(linewidth=220, suppress = True,precision = 7)

#f = open('rks1.out','a')
#Input Section
n = 10
inc = 0.1
#for i in range(-n,2*n):
for i in range(-10,-9):
	#print 0.1*i

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

#z = '''
#O
#H  1  0.9870
#H  1  0.9739  2  109.01
#O  2  1.7958  1  172.27  3  156.75
#H  4  0.9745  2  113.07  1  83.29
#H  4  0.9747  2  109.58  1  320.76
#               '''

	a = z2cart(z)
	print "H bond Length (AA)",z2
	bas = ['6-31g','6-31g']
	xc = ['b3lyp','lda']
	# End of Input



	bo1 = BORKS(a,bas,3,xc)
#	f.write(z2+" "+str(bo1.EBO)+ " " + str(bo1.rks3.e_tot) + "\n")


#f.close()
