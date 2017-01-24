"""
These routines should eventually be removed...
But for the time being it's okay.
"""

import numpy as np

def TransMat(M,U,inv = 1):
    if inv == 1:
        # U.t() * M * U
        Mtilde = np.dot(np.dot(U.T,M),U)
    elif inv == -1:
        # U * M * U.t()
        Mtilde = np.dot(np.dot(U,M),U.T)
    return Mtilde

def TrDot(A,B):
    C = np.trace(np.dot(A,B))
    return C

def MatrixPower(A,p,PrintCondition=False):
	''' Raise a Hermitian Matrix to a possibly fractional power. '''
	#w,v=np.linalg.eig(A)
	# Use SVD
	u,s,v = np.linalg.svd(A)
	if (PrintCondition):
		print "MatrixPower: Minimal Eigenvalue =", np.min(s)
	for i in range(len(s)):
		if (abs(s[i]) < np.power(10.0,-14.0)):
			s[i] = np.power(10.0,-14.0)
	#print("Matrixpower?",np.dot(np.dot(v,np.diag(w)),v.T), A)
	#return np.dot(np.dot(v,np.diag(np.power(w,p))),v.T)
	return np.dot(u,np.dot(np.diag(np.power(s,p)),v))
