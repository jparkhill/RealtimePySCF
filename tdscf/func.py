"""
These routines should eventually be removed...
But for the time being it's okay.
"""

import numpy as np
import os,sys
import ctypes
from pyscf import lib

OCCDROP = 1e-12
BLKSIZE = 96



def load_library(libname):
# numpy 1.6 has bug in ctypeslib.load_library, see numpy/distutils/misc_util.py
    if '1.6' in np.__version__:
        if (sys.platform.startswith('linux') or
            sys.platform.startswith('gnukfreebsd')):
            so_ext = '.so'
        elif sys.platform.startswith('darwin'):
            so_ext = '.dylib'
        elif sys.platform.startswith('win'):
            so_ext = '.dll'
        else:
            raise OSError('Unknown platform')
        libname_so = libname + so_ext
        return ctypes.CDLL(os.path.join(os.path.dirname(__file__), libname_so))
    else:
        _loaderpath = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, _loaderpath)

libtdscf = load_library( os.path.expanduser('~') +'/tdscf_pyscf/lib/tdscf/libtdscf')
libdft = lib.load_library('libdft')
libcvhf = lib.load_library('libcvhf')
def eval_rhoc(mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
              verbose=None):

    assert(ao.flags.c_contiguous)
    xctype = xctype.upper()
    if xctype == 'LDA':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = np.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=np.int8)
    pos = mo_occ.real > OCCDROP
    cpos = np.einsum('ij,j->ij', mo_coeff[:,pos], np.sqrt(mo_occ[pos]))
    if pos.sum() > 0:
        if xctype == 'LDA':
            c0 = Z_dot_ao_dm(mol, ao, cpos, nao, ngrids, non0tab)
            #c0 = _dot_ao_dm(mol, ao.real, cpos.real, nao, ngrids, non0tab) #c_dot_product(ao,cpos)
            #c0 -= _dot_ao_dm(mol, ao.imag, cpos.imag, nao, ngrids, non0tab)
            rho = np.einsum('pi,pi->p', c0, c0.conj())
        elif xctype == 'GGA':
            rho = np.empty((4,ngrids)).astype(complex)
            #c0 = c_dot_product(ao[0],cpos)
            c0 = Z_dot_ao_dm(mol, ao[0], cpos, nao, ngrids, non0tab)
            # c0 = _dot_ao_dm(mol, ao[0].real, cpos.real, nao, ngrids, non0tab)
            # c0 -= _dot_ao_dm(mol, ao[0].imag, cpos.imag, nao, ngrids, non0tab)
            rho[0] = np.einsum('pi,pi->p', c0, c0.conj())
            for i in range(1, 4):
                #c1 = c_dot_product(ao[i],cpos)
                c1 = Z_dot_ao_dm(mol, ao[i], cpos, nao, ngrids, non0tab)
                #c1 = _dot_ao_dm(mol, ao[i].real, cpos.real, nao, ngrids, non0tab)
                #c1 -= _dot_ao_dm(mol, ao[i].imag, cpos.imag, nao, ngrids, non0tab)
                rho[i] = np.einsum('pi,pi->p', c0, c1.conj()) * 2 # *2 for +c.c.
        else: # meta-GGA
            # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
            rho = np.empty((6,ngrids))
            c0 = Z_dot_ao_dm(mol, ao[0], cpos, nao, ngrids, non0tab)
            # c0 = _dot_ao_dm(mol, ao[0].real, cpos.real, nao, ngrids, non0tab) #c_dot_product(ao[0],cpos)
            # c0 -= _dot_ao_dm(mol, ao[0].imag, cpos.imag, nao, ngrids, non0tab)
            rho[0] = np.einsum('pi,pi->p', c0, c0.conj())
            rho[5] = 0
            for i in range(1, 4):
                c1 = Z_dot_ao_dm(mol, ao[i], cpos, nao, ngrids, non0tab)
                # c1 = _dot_ao_dm(mol, ao[i].real, cpos.real, nao, ngrids, non0tab)#c_dot_product(ao[i],cpos)
                # c1 -= _dot_ao_dm(mol, ao[i].imag, cpos.imag, nao, ngrids, non0tab)
                rho[i] = np.einsum('pi,pi->p', c0, c1.conj()) * 2 # *2 for +c.c.
                rho[5] += np.einsum('pi,pi->p', c1, c1.conj())
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            c1 = Z_dot_ao_dm(mol, ao2.real, cpos.real, nao, ngrids, non0tab)
            # c1 = _dot_ao_dm(mol, ao2.real, cpos.real, nao, ngrids, non0tab)#c_dot_product(ao[i],cpos)
            # c1 -= _dot_ao_dm(mol, ao2.imag, cpos.imag, nao, ngrids, non0tab)
            rho[4] = np.einsum('pi,pi->p', c0, c1.conj())
            rho[4] += rho[5]
            rho[4] *= 2

            rho[5] *= .5
    else:
        if xctype == 'LDA':
            rho = np.zeros(ngrids)
        elif xctype == 'GGA':
            rho = np.zeros((4,ngrids))
        else:
            rho = np.zeros((6,ngrids))

    neg = mo_occ.real < -OCCDROP
    if neg.sum() > 0:
        cneg = np.einsum('ij,j->ij', mo_coeff[:,neg], np.sqrt(-mo_occ[neg]))
        if xctype == 'LDA':
            c0 = Z_dot_ao_dm(mol, ao, cneg, nao, ngrids, non0tab)
            # c0 = _dot_ao_dm(mol, ao.real, cneg.real, nao, ngrids, non0tab)
            # c0 -= _dot_ao_dm(mol, ao.imag, cneg.imag, nao, ngrids, non0tab)
            rho -= np.einsum('pi,pi->p', c0, c0.conj())
        elif xctype == 'GGA':
            #c0 = c_dot_product(ao[0],cneg)
            c0 = Z_dot_ao_dm(mol, ao[0], cneg, nao, ngrids, non0tab)
            # c0 = _dot_ao_dm(mol, ao[0].real, cneg.real, nao, ngrids, non0tab)
            # c0 -= _dot_ao_dm(mol, ao[0].imag, cneg.imag, nao, ngrids, non0tab)
            rho[0] -= np.einsum('pi,pi->p', c0, c0.conj())
            for i in range(1, 4):
                #c1 = c_dot_product(ao[i],cneg)
                c1 = Z_dot_ao_dm(mol, ao[i], cneg, nao, ngrids, non0tab)
                # c1 = _dot_ao_dm(mol, ao[i].real, cneg.real, nao, ngrids, non0tab)
                # c1 = _dot_ao_dm(mol, ao[i].imag, cneg.imag, nao, ngrids, non0tab)
                rho[i] -= np.einsum('pi,pi->p', c0, c1.conj()) * 2 # *2 for +c.c.
        else:
            c0 = Z_dot_ao_dm(mol, ao[0], cneg, nao, ngrids, non0tab)
            # c0 = _dot_ao_dm(mol, ao[0].real, cneg.real, nao, ngrids, non0tab)#c_dot_product(ao[0],cneg)
            # c0 -= _dot_ao_dm(mol, ao[0].imag, cneg.imag, nao, ngrids, non0tab)
            rho[0] -= np.einsum('pi,pi->p', c0, c0.conj())
            rho5 = 0
            for i in range(1, 4):
                c1 = Z_dot_ao_dm(mol, ao[i], cneg, nao, ngrids, non0tab)
                # c1 = _dot_ao_dm(mol, ao[i].real, cneg.real, nao, ngrids, non0tab)#c_dot_product(ao[i],cneg)
                # c1 = _dot_ao_dm(mol, ao[i].imag, cneg.imag, nao, ngrids, non0tab)
                rho[i] -= np.einsum('pi,pi->p', c0, c1.conj()) * 2 # *2 for +c.c.
                rho5 -= np.einsum('pi,pi->p', c1, c1.conj())
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            c1 = Z_dot_ao_dm(mol, ao2, cneg, nao, ngrids, non0tab)
            # c1 = _dot_ao_dm(mol, ao2.real, cneg.real, nao, ngrids, non0tab)#c_dot_product(ao[i],cneg)
            # c1 = _dot_ao_dm(mol, ao2.imag, cneg.imag, nao, ngrids, non0tab)
            rho[4] -= np.einsum('pi,pi->p', c0, c1.conj()) * 2
            rho[4] -= rho5 * 2

            rho[5] -= rho5 * .5
    return rho.real

def c_dot_product(A,B):
    '''
    using ARMA
    return A*B; assuming A and B are complex (or force it to be complex)
    '''
    na, nb = A.shape
    nc = B.shape[1]
    a = np.asarray(A, order='C').astype(complex)
    b = np.asarray(B, order='C').astype(complex)
    c = np.zeros((na,nc)).astype(complex)


    libtdscf.i_dot_product(
    a.ctypes.data_as(ctypes.c_void_p),b.ctypes.data_as(ctypes.c_void_p),c.ctypes.data_as(ctypes.c_void_p),\
    ctypes.c_int(na),ctypes.c_int(nb),ctypes.c_int(nc))
    return c

def r_dot_product(A,B):
    '''
    using ARMA
    return A*B; assuming A and B are complex
    '''
    na, nb = A.shape
    nc = B.shape[1]
    a = np.asarray(A, order='C').astype(float)
    b = np.asarray(B, order='C').astype(float)
    c = np.zeros((na,nc)).astype(float)


    libtdscf.r_dot_product(
    a.ctypes.data_as(ctypes.c_void_p),b.ctypes.data_as(ctypes.c_void_p),c.ctypes.data_as(ctypes.c_void_p),\
    ctypes.c_int(na),ctypes.c_int(nb),ctypes.c_int(nc))
    return c

def TransMat(M,U,inv = 1):
    if inv == 1:
        # U.t() * M * U
        Mtilde = np.dot(np.dot(U.T.conj(),M),U)
    elif inv == -1:
        # U * M * U.t()
        Mtilde = np.dot(np.dot(U,M),U.T.conj())
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

def _dot_ao_ao(mol, ao1, ao2, nao, ngrids, non0tab):
    '''return numpy.dot(ao1.T, ao2)'''
    natm = ctypes.c_int(mol._atm.shape[0])
    nbas = ctypes.c_int(mol.nbas)
    ao1 = np.asarray(ao1, order='C')
    ao2 = np.asarray(ao2, order='C')
    vv = np.empty((nao,nao))
    libdft.VXCdot_ao_ao(vv.ctypes.data_as(ctypes.c_void_p),
                        ao1.ctypes.data_as(ctypes.c_void_p),
                        ao2.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao), ctypes.c_int(ngrids),
                        ctypes.c_int(BLKSIZE),
                        non0tab.ctypes.data_as(ctypes.c_void_p),
                        mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                        mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                        mol._env.ctypes.data_as(ctypes.c_void_p))
    return vv


def _dot_ao_dm(mol, ao, dm, nao, ngrids, non0tab):
    '''return numpy.dot(ao, dm) complex(must edit this inside original PYSCF)'''
    natm = ctypes.c_int(mol._atm.shape[0])
    nbas = ctypes.c_int(mol.nbas)
    vm = np.empty((ngrids,dm.shape[1]))
    ao = np.asarray(ao, order='C')
    dm = np.asarray(dm, order='C')
    libdft.VXCdot_ao_dm(vm.ctypes.data_as(ctypes.c_void_p),
                        ao.ctypes.data_as(ctypes.c_void_p),
                        dm.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao), ctypes.c_int(dm.shape[1]),
                        ctypes.c_int(ngrids), ctypes.c_int(BLKSIZE),
                        non0tab.ctypes.data_as(ctypes.c_void_p),
                        mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                        mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                        mol._env.ctypes.data_as(ctypes.c_void_p))
    return vm

def Z_dot_ao_dm(mol, ao, dm, nao, ngrids, non0tab):
    '''return numpy.dot(ao, dm) complex(must edit this inside original PYSCF)'''
    natm = ctypes.c_int(mol._atm.shape[0])
    nbas = ctypes.c_int(mol.nbas)
    vm = np.empty((ngrids,dm.shape[1])).astype(complex)
    ao = np.asarray(ao, order='C').astype(complex)
    dm = np.asarray(dm, order='C').astype(complex)
    libdft.Z_ao_dm(vm.ctypes.data_as(ctypes.c_void_p),
                        ao.ctypes.data_as(ctypes.c_void_p),
                        dm.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao), ctypes.c_int(dm.shape[1]),
                        ctypes.c_int(ngrids), ctypes.c_int(BLKSIZE),
                        non0tab.ctypes.data_as(ctypes.c_void_p),
                        mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                        mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                        mol._env.ctypes.data_as(ctypes.c_void_p))
    return vm

# def _fpointer(name):
#     return ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, name))
#
# def direct(dms, atm, bas, env, vhfopt=None, hermi=0):
#     c_atm = np.asarray(atm, dtype=np.int32, order='C')
#     c_bas = np.asarray(bas, dtype=np.int32, order='C')
#     c_env = np.asarray(env, dtype=np.double, order='C')
#     natm = ctypes.c_int(c_atm.shape[0])
#     nbas = ctypes.c_int(c_bas.shape[0])
#
#     if isinstance(dms, np.ndarray) and dms.ndim == 2:
#         n_dm = 1
#         nao = dms.shape[0]
#         dms = (np.asarray(dms, order='C'),)
#     else:
#         n_dm = len(dms)
#         nao = dms[0].shape[0]
#         dms = np.asarray(dms, order='C')
#
#     if vhfopt is None:
#         cintor = _fpointer('cint2e_sph')
#         cintopt = make_cintopt(c_atm, c_bas, c_env, 'cint2e_sph')
#         cvhfopt = pyscf.lib.c_null_ptr()
#     else:
#         vhfopt.set_dm(dms, atm, bas, env)
#         cvhfopt = vhfopt._this
#         cintopt = vhfopt._cintopt
#         cintor = vhfopt._intor
#
#     fdrv = getattr(libcvhf, 'CC_direct_drv') # change in original PYSCF nr_direct
#     fdot = _fpointer('CC_nrs8') #nr_direct
#     fvj = _fpointer('CVHFnrs8_ji_s2kl') #nr_incore
#     if hermi == 1:
#         fvk = _fpointer('CVHFnrs8_li_s2kj')
#     else:
#         fvk = _fpointer('CVHFnrs8_li_s1kj')
#     vjk = np.empty((2,n_dm,nao,nao))
#     fjk = (ctypes.c_void_p*(2*n_dm))()
#     dmsptr = (ctypes.c_void_p*(2*n_dm))()
#     vjkptr = (ctypes.c_void_p*(2*n_dm))()
#     for i in range(n_dm):
#         dmsptr[i] = dms[i].ctypes.data_as(ctypes.c_void_p)
#         vjkptr[i] = vjk[0,i].ctypes.data_as(ctypes.c_void_p)
#         fjk[i] = fvj
#     for i in range(n_dm):
#         dmsptr[n_dm+i] = dms[i].ctypes.data_as(ctypes.c_void_p)
#         vjkptr[n_dm+i] = vjk[1,i].ctypes.data_as(ctypes.c_void_p)
#         fjk[n_dm+i] = fvk
#     shls_slice = (ctypes.c_int*8)(*([0, c_bas.shape[0]]*4))
#     ao_loc = np.asarray(make_ao_loc(bas), dtype=np.int32)
#
#     fdrv(cintor, fdot, fjk, dmsptr, vjkptr,
#          ctypes.c_int(n_dm*2), ctypes.c_int(1),
#          shls_slice, ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cvhfopt,
#          c_atm.ctypes.data_as(ctypes.c_void_p), natm,
#          c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
#          c_env.ctypes.data_as(ctypes.c_void_p))
#
#     # vj must be symmetric
#     for idm in range(n_dm):
#         vjk[0,idm] = pyscf.lib.hermi_triu(vjk[0,idm], 1)
#     if hermi != 0: # vk depends
#         for idm in range(n_dm):
#             vjk[1,idm] = pyscf.lib.hermi_triu(vjk[1,idm], hermi)
#     if n_dm == 1:
#         vjk = vjk.reshape(2,nao,nao)
#     return vjk
