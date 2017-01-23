#ifndef TCL_Scratch
#define TCL_Scratch 
#include "BSetMgr.hh"
#include "BasisSet.hh"
#include "OneEMtrx.hh"
#include "ShlPrs.hh"
#include "ShlPrMgr.hh"
#include <armadillo>

using namespace arma;

// To Reduce reallocation, especially in the fock build.
// And also in TCL_Matrices.
class TDSCF_Temporaries
{
public:
    int NBas6D, NB2car;
    double *rP,*zP,*rKptr,*zKptr,*vJ,*vPtot;
    mat zPTemp,rPTemp;
    TDSCF_Temporaries()
    {
        int bCode = rem_read(REM_BASIS2);
        BasisSet s1 = BasisSet(bCode);
        ShlPrs s2 = ShlPrs(s1);
        NBas6D = bSetMgr.crntShlsStats(STAT_NBAS6D);
        NB2car = s2.getNB2car();
        rP = QAllocDoubleWithInit(NBas6D*NBas6D);
        zP = QAllocDoubleWithInit(NBas6D*NBas6D);
        rKptr = QAllocDoubleWithInit(NBas6D*NBas6D);
        zKptr = QAllocDoubleWithInit(NBas6D*NBas6D);
        vJ = QAllocDoubleWithInit(NB2car+1);
        vPtot = QAllocDoubleWithInit(NB2car);
    }
    void PrepareIteration(const cx_mat& P)
    {
        int bCode = rem_read(REM_BASIS2);
        BasisSet s1 = BasisSet(bCode);
        ShlPrs s2 = ShlPrs(s1);
        rPTemp = real(P);
        zPTemp = imag(P);
        VRscale(rP,NBas6D*NBas6D,0.0);
        VRscale(zP,NBas6D*NBas6D,0.0);
        VRscale(rKptr,NBas6D*NBas6D,0.0);
        VRscale(zKptr,NBas6D*NBas6D,0.0);
        VRscale(vJ,NB2car+1,0.0);
        VRscale(vPtot,NB2car,0.0);
        memcpy(rP, rPTemp.memptr(),rPTemp.n_elem*sizeof(double));
        memcpy(zP, zPTemp.memptr(),zPTemp.n_elem*sizeof(double));
        ScaV2M(rPTemp.memptr(), vPtot, TRUE, FALSE, s2);
    }
    
    ~TDSCF_Temporaries()
    {
        cout << "Deleting TDSCF temporaries." << endl;
        QFree(rP);
        QFree(zP);
        QFree(rKptr);
        QFree(zKptr);
        QFree(vJ);
        QFree(vPtot);
    }
};

#endif
