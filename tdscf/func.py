"""
These routines should eventually be removed...
But for the time being it's okay. 
"""

import numpy as np

def TransMat(M,U,inv = 1):
    if inv == 1:
        # U.t() * M * U
        Mtilde = np.dot(np.dot(np.transpose(U),M),U)
    elif inv == -1:
        # U * M * U.t()
        Mtilde = np.dot(np.dot(U,M),np.transpose(U))
    return Mtilde

def TrDot(A,B):
    C = np.trace(np.dot(A,B))
    return C
