#include "one-e.h"
#include "math.h"
using namespace std;

/* Yihan, 03/2012
   This function aims at generating the following one-electron matrices: 
      1: Overlap matrix
      2: 1E matrix with Anna's shifted quadratic function
      3: Coulomb Potential from ECP 
   In principle, this can be generated to compute other qualities, such as
      -  three-center overlap
      -  four-center overlap
   which are needed in, for example, Poisson fitting.
*/

// A superposition of gaussians that make an 'ND' in the xy plane.
double ND(double x, double y, double z)
{
    double ND[18*2];
    memset(ND,0.0,38*sizeof(double));
    ND[0] = 0; ND[1] = 0.0;
    ND[2] = 0; ND[3] = 0.7;
    ND[4] = 0; ND[5] = 1.4;
    ND[6] = 0; ND[7] = 2.1;

    ND[8] = 0.7; ND[9] = 1.4;
    ND[10] = 1.4; ND[11] = 0.7;

    ND[12] = 2.1; ND[13] = 0;
    ND[14] = 2.1; ND[15] = 0.7;
    ND[16] = 2.1; ND[17] = 1.4;
    ND[18] = 2.1; ND[19] = 2.1;


    ND[20] = 4.2; ND[21] = 0.7;
    ND[22] = 4.2; ND[23] = 1.4;
    ND[24] = 4.2; ND[25] = 2.1;
    ND[26] = 4.2; ND[27] = 0.0;
    
    ND[34] = 4.9; ND[35] = 0;
    ND[28] = 4.9; ND[29] = 2.1;
    ND[30] = 5.6; ND[31] = 0.7;
    ND[32] = 5.6; ND[33] = 1.4;

    double tore = 0.0;
    for (int i=0; i<36; i+=2)
    {
        ND[i] *= 1.8*(1.1/0.7);
        ND[i+1] *= 1.8*(1.1/0.7);
        tore += exp(-((x-ND[i])*(x-ND[i]) + (y-ND[i+1])*(y-ND[i+1]))/(2.0*0.5*0.5));
    }
    return tore;
}


void NDMatrix(double* jMtrx_x,INTEGER dir, INTEGER jobType, INTEGER grdTyp)
{
    
    //printf("calling OneEMtrxViaNumericalntegration\n");
    // For now,  do not see the need to use other basis sets
    INTEGER IBcode= rem_read(REM_IBASIS);
    SET_QC_FCOMMON(&IBcode);
    
    XCAtoms xcatom;
    
    // This distinguishes it from normal XC jobs
    INTEGER jobID0 = 10+jobType;
    XCFunctional xcFunc;
    XCJobPara xcpara(xcFunc, xcatom, jobID0);
    
    int nAtoms = xcatom.getNAtoms();
    double thresh = xcpara.thresh;
    XCBasisSet basDen(IBcode, nAtoms, thresh);
    xcatom.setSize(basDen);
    
    MoleGrid mgrid(xcatom, grdTyp, xcpara.nDrvNuc, thresh);
    
    tstart();
    
    // (Yihan) do not actually the density matrix
    // but who cares, this is only a pilot code
    INTEGER NBas = bSetMgr.crntShlsStats(STAT_NBASIS);
    double* pDen = QAllocDouble(NBas*NBas);
    FileMan(FM_READ,FILE_DENSITY_MATRIX,FM_DP,NBas*NBas,0,FM_BEG,pDen);
    
    XCOrderedMat denMat(basDen, pDen, xcpara.nDen);
    denMat.toXCMat();
    XCOrderedMat fxcMat(basDen, jMtrx_x, xcpara.nDen);
    
    int nBatch=mgrid.getNBatch();
    
    double xave=0., yave=0., zave=0.;
    int totalgrid=0;
    
    GPI_DLB_reset();
    
    //multiply threads start here
#pragma omp parallel reduction(+:totalgrid,xave,yave,zave)
    {
        
        //local timer: b for basis, v for density building, f for fock building
        GTimer bTimer;
        GTimer vTimer;
        GTimer fTimer;
        
        int ibat=0;
        while ( true )
        {
#pragma omp critical (XCCOUNTER)
            {
                ibat = GPI_DLB_next()-1;
            }
            if (ibat >= nBatch) break;
            
            BatchGrid grid(mgrid, ibat);
            totalgrid+=grid.getNGrid();
            
            BatchShl  sigs1(grid, xcatom, basDen);
            grid.updateDrv(sigs1);
            
            bTimer.start();
            BatchBas  bbas(basDen, xcpara, grid, xcatom, sigs1);
            bTimer.end();
            
            double* jFnlx = QAllocDouble(grid.nGrid);
            {
                // box type potential (cubic if symmetric)
                double * Pts=grid.getPts();
                for(INTEGER iGrid=0; iGrid<grid.nGrid; iGrid++) {
                    double vec[3];
                    for(INTEGER k=0; k<3; k++)
                        vec[k] = Pts[3*iGrid+k];
                    double x = vec[0];
                    double y = vec[1];
                    double z = vec[2];
                    
                    xave += x;
                    yave += y;
                    zave += z;
                    
                    jFnlx[iGrid] = ND(x,y,z);
                }
            }
            
            BatchXCMat xcmat(xcpara, grid, sigs1, bbas);
            xcmat.updateMat(fxcMat, jFnlx, jobID0); // Right now only turning on an x-field.
        }
        
    } // omp parallel
    
    cout << "Mean Grid Point: "  << xave/totalgrid << ","<< yave/totalgrid << ","<< zave/totalgrid << endl;
    
    fxcMat.toHFMat();
    
#ifdef PARALLEL
    int nB2=basDen.nB2;
    GlobalSum(jMtrx_x, nB2, true);
#endif
    tend();
}

void FieldMatricesViaNumericalntegration(double* jMtrx_x,INTEGER dir, INTEGER jobType, INTEGER grdTyp) {
    
    //printf("calling OneEMtrxViaNumericalntegration\n");
    // For now,  do not see the need to use other basis sets
    INTEGER IBcode= rem_read(REM_IBASIS);
    SET_QC_FCOMMON(&IBcode);
    
    XCAtoms xcatom;
    
    // This distinguishes it from normal XC jobs
    INTEGER jobID0 = 10+jobType;
    XCFunctional xcFunc;
    XCJobPara xcpara(xcFunc, xcatom, jobID0);
    
    int nAtoms = xcatom.getNAtoms();
    double thresh = xcpara.thresh;
    XCBasisSet basDen(IBcode, nAtoms, thresh);
    xcatom.setSize(basDen);
    
    MoleGrid mgrid(xcatom, grdTyp, xcpara.nDrvNuc, thresh);
    
    tstart();
    
    // (Yihan) do not actually the density matrix
    // but who cares, this is only a pilot code
    INTEGER NBas = bSetMgr.crntShlsStats(STAT_NBASIS);
    double* pDen = QAllocDouble(NBas*NBas);
    FileMan(FM_READ,FILE_DENSITY_MATRIX,FM_DP,NBas*NBas,0,FM_BEG,pDen);

    XCOrderedMat denMat(basDen, pDen, xcpara.nDen);
    denMat.toXCMat();
    XCOrderedMat fxcMat(basDen, jMtrx_x, xcpara.nDen);
    int nBatch=mgrid.getNBatch();
    int totalgrid=0;

    int *AtNo;
    double *Carts3,*xyz;
    get_carts(NULL,&Carts3,&AtNo,NULL);
    xyz = QAllocDouble(3*nAtoms);
    VRcopy(xyz,Carts3,3*nAtoms);
    //bool useBohr = (rem_read(REM_INPUT_BOHR) == 1) ? true : false;
    //if (!useBohr) VRscale(xyz,3*nAtoms,ConvFac(BOHRS_TO_ANGSTROMS));
    double xm=0;
    double ym=0;
    double zm=0;
    for (int i = 0; i < nAtoms; i++){
        //cout << "Atom #" << i+1 <<". X coord: " << xyz[3*i] << ". Y coord: " << xyz[3*i+1] << ". Z coord: " << xyz[3*i+2] << endl;
        xm += xyz[3*i];
        ym += xyz[3*i+1];
        zm += xyz[3*i+2];
    }
    double origin[3]={xm/nAtoms,ym/nAtoms,zm/nAtoms};
    
    GPI_DLB_reset();
    
    //multiply threads start here
#pragma omp parallel reduction(+:totalgrid)
    {
        //local timer: b for basis, v for density building, f for fock building
        GTimer bTimer;
        GTimer vTimer;
        GTimer fTimer;
        int ibat=0;
        while ( true )
        {
#pragma omp critical (XCCOUNTER)
            {
                ibat = GPI_DLB_next()-1;
            }
            if (ibat >= nBatch) break;
            
            BatchGrid grid(mgrid, ibat);
            totalgrid+=grid.getNGrid();
            
            BatchShl  sigs1(grid, xcatom, basDen);
            grid.updateDrv(sigs1);
            
            bTimer.start();
            BatchBas  bbas(basDen, xcpara, grid, xcatom, sigs1);
            bTimer.end();
            
            double* jFnlx = QAllocDouble(grid.nGrid);
            //compute_shifted_quadratic_functional(jFnl, grid.getPts(), grid.nGrid);
            {
                // box type potential (cubic if symmetric)
                double x0 = 0;
                double y0 = 0;
                double z0 = 0;
                double * Pts=grid.getPts();
                for(INTEGER iGrid=0; iGrid<grid.nGrid; iGrid++) {
                    double vec[3];
                    for(INTEGER k=0; k<3; k++)
                        vec[k] = Pts[3*iGrid+k] - origin[k];
                    double x = vec[0];
                    double y = vec[1];
                    double z = vec[2];
                    
                    jFnlx[iGrid] = 0.0;
                    
                    if(fabs(x) > x0 && dir == 0)
                    {
                        jFnlx[iGrid] += x;
                    }
                    if(fabs(y) > y0 && dir == 1)
                    {
                        jFnlx[iGrid] += y;
                    }
                    if(fabs(z) > z0 && dir == 2)
                    {
                        jFnlx[iGrid] += z;
                    }
                    
                }
            }

            BatchXCMat xcmat(xcpara, grid, sigs1, bbas);
            xcmat.updateMat(fxcMat, jFnlx, jobID0); // Right now only turning on an x-field.
            
        }
        
    } // omp parallel
    
    fxcMat.toHFMat();
    
#ifdef PARALLEL
    int nB2=basDen.nB2;
    GlobalSum(jMtrx_x, nB2, true);
#endif
    tend();
}

