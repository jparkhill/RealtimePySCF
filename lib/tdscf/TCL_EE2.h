#ifndef TCLEE2h
#define TCLEE2h
// <Loggins> DANGERZONEEE...<\Loggins>
#ifndef ARMA_NO_DEBUG
#define ARMA_NO_DEBUG 1
#endif
#ifndef ARMA_CHECK
#define ARMA_CHECK 0
#endif

// Must have user defined reductions which were a feature of OMP 4.0 in 2013.
#ifdef _OPENMP
#if _OPENMP > 201300
#define HASUDR 1
#else
#define HASUDR 0
#endif
#else
#define HASUDR 0
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <armadillo>
#include "math.h"
#include "rks.h"
#include "TCL.h"
#include "TCL_ActiveSpace.h"
#include "TCL_polematrices.h"
// To grab our 2e integrals.
#include "../cdman/rimp2grad.h"
#include "../cdman/ricispd.h"
//#include "noci_mp2.h"
//#include "KeyVtoM.hh"
#include <liblas/liblas.h>
#include "fileman_util.h"

using namespace arma;

class EE2{
public:
    int n_ao, n_mo, n_occ, n_aux;
    int n,n2,n3,n4;
    int tn,tn2,tn3,tn4;
    
    int no,nv,nact;
    int g0flag;
    
    std::map<std::string,double> params;
    ASParameters* asp;
    TCLMatrices* tcl;
    // Eventually active space could speed up both fock build and the correlated part.
    // For now it will just speed up the correlated part.
    // The setup is done in TCL_Prop.h.
    // Rho is passed/propagated in subspace.
    
    bool B_Built;
    bool InCore;
    double nuclear_energy,Ehf,Epseudo,lambda,Emp2,thresh; // Lambda is an adiabatic parameter to warm the system.
    cube BpqR; // Lowdin basis, calculated once on construction and fixed throughout.
    
    mat RSinv;
    mat H,X;
    cx_mat F,V,C;
    cx_mat r2; // The two-body DM.
    cx_mat rho0; // The original density for the inhomogeneity.
    vec eigs;
    
    // This is handy to reduce B rebuilds, if V=Curr then CurrInt is not rebuilt
    cx_mat IntCur, VCur;
    cx_mat Vso;// Spin-orbital V.
    
    typedef complex<double> cx;
    cx j; // The imaginary unit.
    
    EE2(int n_mo_, int n_occ_, rks* the_scf, cx_mat& Rho_, cx_mat& V_,const std::map<std::string,double>& params_ = std::map<std::string,double>(), ASParameters* asp_= NULL, TCLMatrices* tcl_=NULL, FieldMatrices* Mus = NULL): n_mo(n_mo_), j(0.0,1.0), n_occ(n_occ_), H(the_scf->H), X(the_scf->X), V(V_), B_Built(false), params(params_), lambda(1.0), thresh(pow(10.0,-11.0)), asp(asp_), tcl(tcl_), rho0(1,1)
    {
        int LenV = megtot();
        int IBASIS = rem_read(REM_IBASIS);
        int stat = STAT_NBASIS;
        int Job;
        int IBasisAux = rem_read(REM_AUX_BASIS);
        nuclear_energy = the_scf->nuclear_energy();
        
        ftnshlsstats(&n_aux,&stat, &IBasisAux);
        cout << "------------------------------" << endl;
        cout << "-------- EE2 JAP 2015 --------" << endl;
        cout << "--------   Corr " << params["Corr"] << "    --------" << endl;
        cout << "------------------------------" << endl;
        cout << "BBGKY" << params["BBGKY"] << endl; // Does cumulant decomposed BBGKY2
        cout << "RIFock:" << params["RIFock"] << endl;
        
        if (params["ActiveSpace"])
        {
            nact = asp->ActiveLast-asp->ActiveFirst+1;
            cout << "Active Space: " << asp->ActiveFirst << " to " << asp->ActiveLast << " nact " << nact << endl;
        }
#ifdef _OPENMP
        if (omp_get_max_threads() <=1)
            omp_set_num_threads(8);
        cout << "OMP spec:" << _OPENMP << " n_threads " << omp_get_max_threads() << endl;
#if HASUDR
        cout << "Your compiler supports parallism in EE2. You'll be toasty warm soon." << endl;
#else
        cout << "Your compiler does not support OMP parallism in EE2, please recompile with GCC 5.0 or Intel 16.0 or later" << endl;
#endif
#endif
        
        if (params["ActiveSpace"])
        {
            n=nact;
            n2=n*n;
            n3=n2*n;
            n4=n3*n;
            
            tn=2*n;
            tn2=4*n2;
            tn3=8*n3;
            tn4=16*n4;
            
            no=asp->ne;
            nv=n_mo-n_occ;
        }
        else
        {
            n=n_mo;
            n2=n_mo*n_mo;
            n3=n2*n_mo;
            n4=n3*n_mo;
            
            tn=2*n_mo;
            tn2=4*n_mo*n_mo;
            tn3=8*n2*n_mo;
            tn4=16*n3*n_mo;
            
            no=n_occ;
            nv=n_mo-n_occ;
        }
        
        cout << "NMO: " << n_mo << " NAux: " << n_aux << " Size BpqR " << n_mo*n_mo*n_aux*8/(1024*1024) << "MB" << endl;
        cout << "Size IntCur: " << n4*sizeof(cx)/(1024*1024) << "MB" << endl;
        InCore = (n4*sizeof(cx)/(1024*1024) < 10.0); // Whether to do in-core integrals
        if (!InCore)
            cout << "Using a direct algorithm... " << endl;
        
        BpqR.resize(n_mo,n_mo,n_aux);
        RSinv.resize(n_aux,n_aux);
        if(!params["RIFock"])
            IntCur.resize(n2,n2);
        {
            // get (P|Q)**(-1/2) and save it to disk
            //
            //  First, fool trnjob into thinking we want RIMP2
            int tmp1 = rem_read(REM_LEVCOR);
            rem_write(103, REM_LEVCOR);
            trnjob(&Job);
            //  Then, write it back
            rem_write(tmp1, REM_LEVCOR);
            // form (P|Q)^-1/2 matrix and saves it to FILE_2C2E_INV_INTS
            //
            
            threading_policy::enable_blas_only();
            //            invsqrtpq(&n_aux, &Job);
            ISqrtPQ();
            threading_policy::pop();
            
            // read (P|Q)^-1/2
            FileMan(FM_OPEN_RW, FILE_2C2E_INVSQ_INTS, 0, 0, 0, 0, 0);
            LongFileMan(FM_READ, FILE_2C2E_INVSQ_INTS, FM_DP, n_aux*n_aux, 0, FM_BEG, RSinv.memptr(), 0);
            if (params["Print"])
                RSinv.print("(R|S)^-1/2");
            FileMan(FM_CLOSE, FILE_2C2E_INVSQ_INTS, 0, 0, 0, 0, 0);
        }
        //RSinv.print("RSinv");
        BuildB();
        
        if (!params["StartFromSetman"])
        {
            Rho_.eye();
            if (!params["ActiveSpace"])
                Rho_.submat(n_occ,n_occ,n_mo-1,n_mo-1) *= 0.0;
            else
                Rho_.submat(asp->ne,asp->ne,n-1,n-1) *= 0.0;
        }
        FockBuild(Rho_,V_);
        
        if (params["Print"])
            cout << "Initial V build." << endl;
        
        if (!params["RIFock"])
            UpdateVi(V_);
        
        if (params["Print"])
            cout << "Finished initial V build." << endl;
        
        // To run external to Q-Chem and avoid the two minute link.
        if (params["DumpEE2"])
        {
            V_.save("V",arma_binary);
            X.save("X",arma_binary);
            BpqR.save("BpqR",arma_binary);
        }
        
        if (params["BBGKY"])
        {
            // Initialize the two body density separably.
            Separabler2(Rho_);
        }
        
        // Fock build, integral build and initialize the density.
        if (!params["SkipInitial"] and !params["StartFromSetman"])
        {
            CorrelateInitial(Rho_,V_,Mus);
        }
        else
        {
            cout << "Skipping initialization." << endl;
        }
        
        if(!params["ActiveSpace"])
        {
            rho0 = V*Rho_*(V.t());
            
        }
        else
            rho0 = asp->Vs*Rho_*(asp->Vs.t());
        
        
        if (params["TDCIS"] || params["DoCIS"] || params["TDTDA"])
            CIS();
    }
    
    
    void dojob66( double *U, int *jj )
    {
        double t1_66[3];
        qtime_t t0_66;
        
        t0_66 = QTimerOn();
        
        INTEGER bCodeBra;
        INTEGER bCodeKet;
        
        bCodeBra = DEF_ID;
        bCodeKet = DEF_ID;
        
        AOIntsJobInfo JobInfo(66);
        
        AOints(U,NULL,NULL,NULL,NULL,NULL,NULL,NULL,jj,JobInfo,
               ShlPrs(bCodeBra),ShlPrs(bCodeKet),FALSE);
        
        QTimerOff(t1_66,t0_66);
        
#ifdef DEVELOPMENT
        printf("Job 66 time: %lf.\n", t1_66[0] );
#endif
        
    }
    
    // Shamelessly borrowed from libari/RIHF.C
    void ISqrtPQ( void )
    {
        INTEGER IBasisAux = rem_read(REM_AUX_BASIS);
        
        INTEGER NBasAux;
        
        INTEGER stat = STAT_NBASIS;
        
        ftnshlsstats(&NBasAux,&stat,&IBasisAux);
        
        //        double *Us = QAllocDouble( NBasAux * NBasAux );
        mat us(NBasAux,NBasAux),us2(NBasAux,NBasAux);
        double * Us = us.memptr(); us.zeros();
        double * Us2 = us2.memptr(); us2.zeros();
        //        memset(Us, 0, sizeof(double) * NBasAux * NBasAux );
        //        double *Us2 = QAllocDouble( NBasAux * NBasAux );
        int jj = 0;
        
        dojob66(Us,&jj);
        fillsq(Us,&NBasAux);
        
        // Now calculate the pseudo inverse and write to disk.
        mat U;
        vec s;
        mat V;
        svd(us2,s,V,us);
        
        double smallestEigenvalue = 1e10;
        for( int i = 0; i < NBasAux; i++ )
        {
            if( s(i) < smallestEigenvalue )
                smallestEigenvalue = s(i);
            
            if( s(i) > 1e-8 )
                for( int j = 0; j < NBasAux; j++ )
                    Us[i*NBasAux+j] = Us2[i*NBasAux+j] * (1.0 / sqrt(s(i)));
            else
                for( int j = 0; j < NBasAux; j++ )
                    Us[i*NBasAux+j] = 0;
        }
        
        RSinv = V*us.t();
        FileMan(FM_WRITE, FILE_2C2E_INVSQ_INTS,FM_DP,NBasAux*NBasAux,0,FM_BEG,RSinv.memptr());
    }
    
    // Try to adiabatically turn on the correlation.
    void GML(arma::cx_mat& Rho_, arma::cx_mat& V_)
    {
        cout << "Adiabatically Correlating the system." << endl;
        double dt=0.02; int NGML=300;
        cx_mat RhoM12_(Rho_);
        bool ison=false;
        for(int i=0; i<NGML; ++i)
        {
            lambda=(1.0+i)/((double)NGML);
            step(NULL,NULL,eigs,V,Rho_,RhoM12_,i*dt, dt,ison);
            if (i%10==0)
                cout << "Gell-Man Low Iteration:" << i << " Lambda " << lambda << " ehf:" << Ehf << endl;
        }
    }
    
    // Generate rate matrix for populations
    void Stability(cx_mat& Rho)
    {
        cout << "calculating stability matrix... " << endl;
        cx_mat tore(Rho); tore.eye();
        cx_mat ref(Rho); ref.zeros();
        double dn=0.001;
        RhoDot(Rho,ref,0.0);
        int n=n_mo;
        for (int j=0; j<n; ++j)
        {
            cx_mat tmp(Rho);
            tmp(j,j)+=dn;
            cx_mat tmp2(Rho); tmp2.zeros();
            RhoDot(tmp,tmp2,0.0);
            for (int i=0; i<n; ++i)
            {
                tore(i,j) += (tmp2(i,i)-ref(i,i))/dn;
            }
        }
        tore.print("Rate Matrix:");
        cout << "Rate Trace? " << trace(tore) << endl;
        for (int j=0; j<n; ++j)
            cout << "Normed Probability:" << j << " " << accu(tore.col(j)) << endl;
        cx_mat sstates; cx_vec rates;
        eig_gen(rates, sstates, tore);
        rates.print("Rates:");
        cout << "State sum: "<< accu(sstates.row(n-1)) << endl;
        sstates.row(n-1).print("steady state");
    }
    
    // Obtain the initial state for the dynamics.
    void CorrelateInitial(cx_mat& Rho,cx_mat& V_,FieldMatrices* Mus = NULL)
    {
        cout << "Solving the RI-HF problem to ensure self-consistency." << endl;
        double err=100.0;
        int maxit = 400; int it=1;
        
        if(!params["ActiveSpace"])
            rho0 = V*Rho*(V.t());
        else
            rho0 = asp->Vs * Rho * asp->Vs.t();
        
        rho0.print("Inintial Rho");

        if(params["FSCFT"]>0.0)
        {
            double FSlimit = floor(params["FSCFT"]);
            bool tempo;
            cx_double xsum,ysum,zsum;
            mat XA(X); XA.zeros();
            cx_mat XX(X,XA);
            Mus->update(XX);
            XX.zeros();
            for(int i = 0; i < FSlimit; i++) // E strength
                for(int j = 0; j < 3; j++) // Direction
                {
                    double FS=(i+1) * 0.0002;
                    err=100.0;
                    while (abs(err) > pow(10.0,-10) && it<maxit)
                    {
                        
                        cx_mat Rot = FockBuild(Rho,V_, false, Mus, &FS, &j);
                        Rho = Rot.t()*Rho*Rot;
                        Rho.diag() *= 0.0;
                        err=accu(abs(Rho));
                        Rho.eye();
                        if (!params["ActiveSpace"])
                            Rho.submat(n_occ,n_occ,n_mo-1,n_mo-1)*=0.0;
                        else
                            Rho.submat(asp->ne,asp->ne,n-1,n-1) *= 0.0;
                        it++;
                        //F.print("Fock Matrix with Field");
                    }
                    cx_mat Rhol = V_*Rho*(V_.t());
                    //cout << endl <<  "Field Strength: " << FS << endl;
                    if (j == 0)
                    {
                        //cout << "DPx: " << trace(Mus->muxo*Rhol) << endl;
                        xsum += trace(Mus->muxo*Rhol)/FS;
                    }
                    if (j == 1)
                    {
                        //cout << "DPy: " << trace(Mus->muyo*Rhol) << endl;
                        ysum += trace(Mus->muyo*Rhol)/FS;
                    }
                    if (j == 2)
                    {
                        //cout << "DPz: " << trace(Mus->muzo*Rhol) << endl;
                        zsum += trace(Mus->muzo*Rhol)/FS;
                    }
                }
            cout << endl;
            cout << "Ave. Polx(a.u.): " << xsum / FSlimit << endl;
            cout << "Ave. Poly(a.u.): " << ysum / FSlimit << endl;
            cout << "Ave. Polz(a.u.): " << zsum / FSlimit << endl;
            cout << "Ave. PolT(a.u.): " << sqrt(xsum*xsum+ysum*ysum+zsum*zsum)/ FSlimit << endl<<endl;
            
            cout << "quiting on here" << endl;
            exit(0);
        }
        
        err=100.0;it=1;
        while (abs(err) > pow(10.0,-10) && it<maxit)
        {
            cx_mat Rot = FockBuild(Rho,V_);
            Rho = Rot.t()*Rho*Rot;
            Rho.diag() *= 0.0;
            err=accu(abs(Rho));
            Rho.eye(); 
            if (!params["ActiveSpace"])
                Rho.submat(n_occ,n_occ,n_mo-1,n_mo-1)*=0.0;
            else
                Rho.submat(asp->ne,asp->ne,n-1,n-1) *= 0.0;
            if (it%10==0)
                cout << " Fock-Roothan: " << it << std::setprecision(9) << " Ehf" << Ehf << endl;
            it++;
        }
        
        // Various options for non-idempotent initial state:
        // gell-man-low, fermi-dirac, and mp2NO.
        cx_mat Pmp2=MP2(Rho); // Won't actually solve for Pmp2 unless we want it. Otherwise just gets the CE.
        if ((bool)params["GML"] && !params["Mp2NOGuess"])
            GML(Rho,V_);
        lambda = 1.0;
        if (params["Mp2NOGuess"])
        {
            cout << "Generating MP2 1 particle density matrix..." << endl;
            if (params["Mp2NOGuess"]==2.0)
                Rho = diagmat(Pmp2);
            else
                Rho = Pmp2;
        }
        if (params["FD"])
        {
            vec pops(eigs.n_elem);
            double EvPerAu=27.2113;
            double Kb = 8.61734315e-5/EvPerAu;
            double Beta = 1.0/(params["Temp"]*Kb);
            cout << "Finding " << n_occ*2 << " Fermions at Beta " << Beta << endl;
            double mu = eigs(n_occ);
            
            if (params["ActiveSpace"])
            {
                cout << "fd not working with AS yet." << endl;
                throw;
            }
            
            mu -= 0.010;
            pops = 1.0/(1.0+exp(Beta*(eigs-mu)));
            double sm = sum(pops);
            mu += 0.020;
            pops = 1.0/(1.0+exp(Beta*(eigs-mu)));
            double sp = sum(pops);
            // Linear fit for mu
            mu = eigs(n_occ)-0.01 + ((0.02)/(sp-sm))*(n_occ - sm);
            pops = 1.0/(1.0+exp(Beta*(eigs-mu)));
            cout << "Populated with " << sum(pops) << " Fermions at mu = " << mu <<  endl;
            
            //Repeat
            double dmu=0.001;
            
            while(abs(accu(pops)-n_occ)>pow(10.0,-10.0))
            {
                mu -= dmu;
                pops = 1.0/(1.0+exp(Beta*(eigs-mu)));
                sm = sum(pops);
                mu += 2*dmu;
                pops = 1.0/(1.0+exp(Beta*(eigs-mu)));
                sp = sum(pops);
                // Linear fit for mu
                mu = mu-2.0*dmu + ((2.0*dmu)/(sp-sm))*(n_occ - sm);
                pops = 1.0/(1.0+exp(Beta*(eigs-mu)));
                cout << "Populated with " << accu(pops) << " Fermions at mu = " << mu <<  endl;
                if ( abs(sum(pops)-n_occ) < pow(10.0,-10.0))
                    break;
                dmu /= max(min(10.0,abs(((2.0*dmu)/(sp-sm))*(n_occ - sm))),1.1);
            }
            
            Rho.diag().zeros();
            for (int i=0; i<n_mo; ++i)
                if( abs(pops(i)) > pow(10.0,-16.0))
                    Rho(i,i) = pops(i);
            cout << "Populated with " << sum(pops) << " Fermions at mu = " << mu <<  endl;
            Rho.diag().st().print("Rho(fd)");
        }
        if (params["InitialCondition"]==1)
        {
            if (!params["ActiveSpace"])
            {
                cx tmp = Rho(n_occ-1,n_occ-1);
                Rho(n_occ,n_occ)+=tmp;
                Rho(n_occ-1,n_occ-1)*=0.0;
            }
            else
            {
                cx tmp = Rho(asp->ne-1,asp->ne-1);
                Rho(asp->ne,asp->ne)+=tmp;
                Rho(asp->ne-1,asp->ne-1)*=0.0;
            }
        }
        cout << "Initial Trace: " << trace(Rho) << endl;
        eigs.st().print("Eigs Before Rebuild");
        lambda = 1.0;
        FockBuild(Rho,V_);
        if (!params["RIFock"])
            UpdateVi(V_);
        eigs.st().print("Eigs After Rebuild");
        
        if (!params["ActiveSpace"] && !params["RIFock"])
            DebugRhoDot(Rho);
        
        return;
    }
    
    // Print the initial Step, Density, Noccs, and DNoccs.
    void DebugRhoDot(const cx_mat& Rho)
    {
        if (params["ActiveSpace"] || params["RIFock"])
            return;
        Rho.diag().st().print("Inital Rho: ");
        vec noccs; cx_mat u;
        if(!eig_sym(noccs, u, Rho, "std"))
        {
            cout << "Warning... initial density did not diagonalize..." << endl;
            return;
        }
        noccs.st().print("Initial natural occupations");
        
        // Check to see what effect the nonlinearity has...
        cx_mat tmp(Rho),tmp2(Rho),tmp3(Rho); tmp.zeros(); tmp3.zeros();
        tmp2.diag(1)+=0.0001;
        tmp2.diag(-1)+=0.0001;
        RhoDot(tmp2,tmp,1.0);
        tmp2.diag(1)+=0.002;
        tmp2.diag(-1)+=0.002;
        RhoDot(tmp2,tmp3,1.0);
        
        // Obtain the correction to a particle hole excitation energy.
        cout << "-------------------------" << endl;
        cx wks0 = eigs(n_occ)-eigs(n_occ-1);
        cout << "Omega_hf 0: " << wks0 << " " << 27.2113*wks0 << "eV"<< endl;
        cx ehbe(2.0*IntCur((n_occ)*n_mo+(n_occ),(n_occ-1)*n_mo+n_occ-1)-IntCur((n_occ)*n_mo+(n_occ-1),(n_occ)*n_mo+n_occ-1));
        double cr=(imag(tmp(n_occ,n_occ-1)/0.0001));
        cout << "E-H binding energy. " << ehbe << " " << 27.2113*ehbe << "eV"<< endl;
        cout << "GS MP2 Correlation energy. " << Emp2 << " " << 27.2113*Emp2 << "eV"<< endl;
        cout << "CIS Transition energy. " << ehbe << " " << 27.2113*(wks0+ehbe) << "eV"<< endl;
        cout << "EE2 Correction: " << cr << " " << 27.2113*cr<< "eV"<< endl;
        cout << "EE2 Correction (2): " << " " << 27.2113*(imag(tmp3(n_occ,n_occ-1)/0.002))<< "eV"<< endl;
        cout << "-------------------------" << endl;
        cout << " Total Estimated Homo Lumo xsition Energy: " << 27.2113*(cr+wks0+ehbe) << "eV"<< endl;
        cout << " Total Estimated Homo Lumo xsition Energy+MP2: " << 27.2113*(cr+wks0+ehbe+Emp2) << "eV"<< endl;
        cout << "-------------------------" << endl;
        
        vec dnoccs;
        if(!eig_sym(dnoccs, u, Rho+0.02*tmp, "std"))
        {
            cout << "Warning... change in density did not diagonalize..." << endl;
            return;
        }
        dnoccs -= noccs;
        dnoccs.st().print("Initial NO change");
        tmp.print("Initial RhoDot");
    }
    
    void CIS()
    {
        const cx* Vi = IntCur.memptr();
        
        if (params["ActiveSpace"])
        {
            int i = no-1;
            int a = no;
            cout << "Effective HL energy" << eigs(a)-eigs(i)+(2.0*Vi[a*n3+i*n2+i*n+a]-Vi[a*n3+i*n2+a*n+i]);
            return;
        }
        
        cx_mat Hcis(no*nv,no*nv); Hcis.zeros();
        for (int i=0; i<no; ++i)
            for (int ip=0; ip<no; ++ip)
                for (int a=no; a<n; ++a)
                    for (int ap=no; ap<n; ++ap)
                    {
                        if (i==ip&&a==ap)
                            Hcis(ip+no*(ap-no),i+no*(a-no)) += eigs(a)-eigs(i);
                        Hcis(ip+no*(ap-no),i+no*(a-no)) += (2.0*Vi[ap*n3+i*n2+ip*n+a]-Vi[ap*n3+i*n2+a*n+ip]);
                    }
        if (no*nv < 30)
            Hcis.print("Hcis");
        vec eigval; cx_mat Cu;
        eig_sym(eigval,Cu,Hcis);
        eigval*=27.2113;
        eigval.st().print("CIS eigenvalues ");
        
        // Make CIS densities and calculate their Mean-Field energies.
        for (int ii = 0; ii < no*nv; ++ii)
        {
            cout << "CIS Vector " << eigval(ii) << endl;
            cx_vec v = Cu.col(ii);
            v.st().print("Vec");
            cout << "Norm" << dot(v,v) << endl;
            cx_vec v2=Hcis*v;
            cout << "vHv" << dot(v,v2) << endl;
            
            // Construct the density.
            cx_mat Rho0(n,n); Rho0.eye(); Rho0 *=2.0;
            Rho0.submat(no,no,n-1,n-1).zeros();
            for (int i=0; i<no; ++i)
                for (int ip=0; ip<no; ++ip)
                    for (int a=no; a<n; ++a)
                        Rho0(i,ip) -= v(no*(a-no)+i)*v(no*(a-no)+ip);
            
            for (int i=0; i<no; ++i)
                for (int a=no; a<n; ++a)
                    for (int ap=no; ap<n; ++ap)
                        Rho0(a,ap) += v(no*(a-no)+i)*v(no*(ap-no)+i);
            
            Rho0*=0.5;
            cout << "MF energy: " << MeanFieldEnergy(Rho0) << endl;
            Rho0.diag().st().print("Diag Pops: ");
            
            if (params["DoCISD"])
               CISD3(eigval(ii) , v);
            
        }
        
        
        
        return;
    }

    
    void CISD(const double e, cx_vec v)
    {
        BuildSpinOrbitalV2();
        // Make a spin orbital version of the amplitude too.
        cx_mat bai(2*nv,2*no); bai.zeros();
        for (int i=0; i<2*no; ++i)
            for (int a=0; a<2*nv; ++a)
            {
                // Only singlets.
                if ( (i < no) && !(a<nv) || !(i < no) && (a<nv))
                    continue;
                bai(a,i) = v(no*(a%nv)+(i%no));
            }
        
        bai /= sqrt(dot(bai,bai));
        
        double dE1=0.0;
        double dE2=0.0;
        double dE3=0.0;
        double dE4=0.0;
        const cx* Vi = Vso.memptr();
        
        int tno = 2*no;
        int tnv = 2*nv;
        
        cx_mat u(4.0*nv*nv,4.0*no*no); u.zeros();
        cx_mat Rab(2.0*nv,2.0*nv); Rab.zeros();
        cx_mat Rij(2.0*no,2.0*no); Rij.zeros();
        cx_mat wai(2.0*nv,2.0*no); wai.zeros();
        cx_mat T(4.0*nv*nv,4.0*no*no); T.zeros();
        
        for (int i=0; i<2*no; ++i)
            for (int j=0; j<2*no; ++j)
                for (int a=0; a<2*nv; ++a)
                    for (int b=0; b<2*nv; ++b)
                    {
                        T(a*tnv+b,i*tno+j) -= real((Vi[(a+tno)*tn3+(b+tno)*tn2+i*tn+j]-Vi[(a+tno)*tn3+(b+tno)*tn2+j*tn+i])/(eigs((a%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((j%no))));
                    }
        
        // Check the MP2 energy
        double TestMp2=0.0;
        for (int i=0; i<2*no; ++i)
            for (int j=0; j<2*no; ++j)
                for (int a=0; a<2*nv; ++a)
                    for (int b=0; b<2*nv; ++b)
                    {
                        TestMp2 += real(0.25*(Vi[(a+tno)*tn3+(b+tno)*tn2+i*tn+j]-Vi[(a+tno)*tn3+(b+tno)*tn2+j*tn+i])*T(a*2*nv+b,i*2*no+j));
                    }
        
        
        //Term1:
        for (int i=0; i<2*no; ++i)
            for (int j=0; j<2*no; ++j)
                for (int a=0; a<2*nv; ++a)
                    for (int b=0; b<2*nv; ++b)
                        for (int k=0; k<2*no; ++k)
                        {
                            u(a*2*nv+b,i*2*no+j) += (Vi[k*tn3+(a+tno)*tn2+i*tn+j]-Vi[k*tn3+(a+tno)*tn2+j*tn+i])*bai(b,k);
                            u(a*2*nv+b,i*2*no+j) -= (Vi[k*tn3+(b+tno)*tn2+i*tn+j]-Vi[k*tn3+(b+tno)*tn2+j*tn+i])*bai(a,k);
                        }
        
        for (int i=0; i<tno; ++i)
            for (int j=0; j<tno; ++j)
                for (int a=0; a<tnv; ++a)
                    for (int b=0; b<tnv; ++b)
                        for (int c=0; c<tnv; ++c)
                        {
                            u(a*tnv+b,i*tno+j) += (Vi[(a+tno)*tn3+(b+tno)*tn2+(c+tno)*tn+j]-Vi[(a+tno)*tn3+(b+tno)*tn2+j*tn+(c+tno)])*bai(c,i);
                            u(a*tnv+b,i*tno+j) -= (Vi[(a+tno)*tn3+(b+tno)*tn2+(c+tno)*tn+i]-Vi[(a+tno)*tn3+(b+tno)*tn2+i*tn+(c+tno)])*bai(c,j);
                        }

        for (int i=0; i<tno; ++i)
            for (int j=0; j<tno; ++j)
                for (int a=0; a<tnv; ++a)
                    for (int b=0; b<tnv; ++b)
                    {
                        dE1 -= 0.25*real(conj(u(a*tnv+b,i*tno+j))*u(a*tnv+b,i*tno+j)/(eigs((a%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((j%no)) - e/27.2113));
                    }
        
        
        //Term2: 
        for (int j=0; j<2*no; ++j)
            for (int b=0; b<2*nv; ++b)
            	for (int c=0; c<2*nv; ++c)
                    for (int k=0; k<2*no; ++k)
                        for (int a=0; a<tnv; ++a)
                            Rab(a,b) += 0.5*(Vi[(j)*tn3+(k)*tn2+(b+tno)*tn+(c+tno)] - Vi[(j)*tn3+(k)*tn2+(c+tno)*tn+(b+tno)])*T(c*tnv+a,j*tno+k); // taken from John's dE2, hermitian

        for (int a=0; a<2*nv; ++a)
            for (int b=0; b<2*nv; ++b)
                for (int i=0; i<tno; ++i)
                    dE2 += real(bai(a,i)*bai(b,i)*Rab(a,b));

	//real(bai(a,i)*bai(b,i)*Rab(a,b)) = real(delta(i,j)*bai(a,i)*bai(b,j)*Rab(a,b)) = real(delta(i,j)*bai(a,j)*bai(b,i)*Rab(a,b));


        //Term3:
        for (int j=0; j<2*no; ++j)
            for (int a=0; a<2*nv; ++a)
                for (int b=0; b<2*nv; ++b)
                    for (int k=0; k<2*no; ++k)
                        for (int i=0; i<tno; ++i)
                            Rij(i,j) += 0.5*(Vi[(j)*tn3+(k)*tn2+(b+tno)*tn+(a+tno)] - Vi[(j)*tn3+(k)*tn2+(a+tno)*tn+(b+tno)])*T(a*tnv+b,i*tno+k); // taken from John's dE3, hermitian

        for (int i=0; i<tno; ++i)
            for (int j=0; j<2*no; ++j)
                for (int c=0; c<tnv; ++c)
                    dE3 += real(bai(c,i)*bai(c,j)*Rij(i,j));

	//real(bai(c,i)*bai(c,j)*Rij(i,j)) = real(delta(c,a)*bai(a,i)*bai(c,j)*Rij(i,j)) = real(delta(c,a)*bai(c,i)*bai(a,j)*Rij(i,j));
/*
        double dE3check=0.0;
        for (int i=0; i<tno; ++i)
            for (int j=0; j<2*no; ++j)
                for (int c=0; c<tnv; ++c)
		    for (int a=0; a<tnv; ++a)
                        dE3check += real(delta(c,a)*bai(a,i)*bai(c,j)*Rij(i,j));

        double dE3check2=0.0;
        for (int i=0; i<tno; ++i)
            for (int j=0; j<2*no; ++j)
                for (int c=0; c<tnv; ++c)
                    for (int a=0; a<tno; ++a)
                        dE3check2 += real(delta(c,a)*bai(c,i)*bai(a,j)*Rij(i,j));
*/

        //Term4:
        cx_mat wai1(4.0*nv*no,4.0*nv*no); wai1.zeros(); // wai1 should have dimension of ia jb
        for (int i=0; i<tno; ++i)
            for (int j=0; j<tno; ++j)
                for (int a=0; a<tnv; ++a)
                    for (int b=0; b<tnv; ++b)
                        for (int k=0; k<tno; ++k)
                            for (int c=0; c<tnv; ++c)
                                wai1(i*tnv+a,j*tnv+b) += (Vi[j*tn3+k*tn2+(b+tno)*tn+(c+tno)]-Vi[j*tn3+k*tn2+(c+tno)*tn+(b+tno)])*T(a*tnv+c,i*tno+k);// hermitian
	
        for (int i=0; i<tno; ++i)
            for (int j=0; j<tno; ++j)
                for (int a=0; a<tnv; ++a)
                    for (int b=0; b<tnv; ++b)
                        wai(a,i) += wai1(i*tnv+a,j*tnv+b)*bai(b,j);

	for (int i=0; i<tno; ++i)
            for (int a=0; a<tnv; ++a)
                dE4 += real(bai(a,i)*wai(a,i));

        cout << "Term1 - Ex State Corr:    " << dE1 << endl;
        cout << "Term2 - OOV:              " << dE2 << endl;
        cout << "Term3 - VVO:              " << dE3 << endl;
        cout << "Term4 - OV:               " << dE4 << endl;
        cout << "*Total CIS(D) correction: " << (dE1+dE2+dE3+dE4) << " au -- " << (dE1+dE2+dE3+dE4)*27.2113 << " eV" << endl;
        cout << "*Mp2:                     " << Emp2 << endl;
        cout << "*Total:                   " << e+(dE1+dE2+dE3+dE4)*27.2113 << " eV" << endl;
    }


     void CISD2(const double ee, cx_vec v)
    {
        BuildSpinOrbitalV2();
        // Make a spin orbital version of the amplitude too.
        cx_mat bai(2*nv,2*no); bai.zeros();
        for (int i=0; i<2*no; ++i)
            for (int a=0; a<2*nv; ++a)
            {
                // Only singlets.
                if ( (i < no) && !(a<nv) || !(i < no) && (a<nv))
                    continue;
                bai(a,i) = v(no*(a%nv)+(i%no));
            }

        bai /= sqrt(dot(bai,bai));

        double dE1=0.0;
        double dE2=0.0;
        double dE3=0.0;
        double dE4=0.0;
        double e1=0.0;
        double e2=0.0;
        double e3=0.0;
        double e4=0.0;
        double e5=0.0;
        double e6=0.0;
        double e7=0.0;
        double e8=0.0;
        double e9=0.0;
        double e10=0.0;
        double e11=0.0;
        double e12=0.0;
        double e13=0.0;
        double e14=0.0;
        double e15=0.0;
        double e16=0.0;
	
        const cx* Vi = Vso.memptr();

        int tno = 2*no;
        int tnv = 2*nv;

	cx_mat Rab(2.0*nv,2.0*nv); Rab.zeros();
        cx_mat Rij(2.0*no,2.0*no); Rij.zeros();
        cx_mat wai(2.0*nv,2.0*no); wai.zeros();
        cx_mat wai1(4.0*nv*no,4.0*nv*no); wai1.zeros(); 
        cx_mat T(4.0*nv*nv,4.0*no*no); T.zeros();
        
        for (int i=0; i<2*no; ++i)
            for (int j=0; j<2*no; ++j)
                for (int a=0; a<2*nv; ++a)
                    for (int b=0; b<2*nv; ++b)
                    {
                        T(a*tnv+b,i*tno+j) -= real((Vi[(a+tno)*tn3+(b+tno)*tn2+i*tn+j]-Vi[(a+tno)*tn3+(b+tno)*tn2+j*tn+i])/(eigs((a%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((j%no))));
                    }
        
        // Check the MP2 energy
        double TestMp2=0.0;
        for (int i=0; i<2*no; ++i)
            for (int j=0; j<2*no; ++j)
                for (int a=0; a<2*nv; ++a)
                    for (int b=0; b<2*nv; ++b)
                    {
                        TestMp2 += real(0.25*(Vi[(a+tno)*tn3+(b+tno)*tn2+i*tn+j]-Vi[(a+tno)*tn3+(b+tno)*tn2+j*tn+i])*T(a*2*nv+b,i*2*no+j));
                    }
        
        
        //Term1:
/*
	U1 = {{S,1},{{\[Delta],1},{i},{j}},{{bt,1},{b},{j}},{{Vt,2},{d,e},{b,m}},{{bs,1},{a},{i}},{{Vs,2},{d,e},{a,m}}}
	U2 = {{S,-1},{{bt,1},{b},{j}},{{Vt,2},{d,e},{b,i}},{{bs,1},{a},{i}},{{Vs,2},{d,e},{a,j}}}
	U3 = {{S,1},{{bt,1},{b},{j}},{{Vt,2},{d,a},{b,m}},{{bs,1},{a},{i}},{{Vs,2},{i,d},{j,m}}}
	U4 = {{S,-1},{{bt,1},{b},{j}},{{Vt,2},{a,e},{b,m}},{{bs,1},{a},{i}},{{Vs,2},{i,e},{j,m}}}
	U5 = U2
	U6 = {{S,1},{{\[Delta],1},{i},{j}},{{bt,1},{b},{j}},{{Vt,2},{d,e},{b,l}},{{bs,1},{a},{i}},{{Vs,2},{d,e},{a,l}}}
	U7 = {{S,-1},{{bt,1},{b},{j}},{{Vt,2},{d,a},{b,l}},{{bs,1},{a},{i}},{{Vs,2},{i,d},{l,j}}}
	U8 = {{S,1},{{bt,1},{b},{j}},{{Vt,2},{a,e},{b,l}},{{bs,1},{a},{i}},{{Vs,2},{i,e},{l,j}}}
	U9 = {{S,1},{{bt,1},{b},{j}},{{Vt,2},{j,d},{i,m}},{{bs,1},{a},{i}},{{Vs,2},{d,b},{a,m}}}
	U10 = {{S,-1},{{bt,1},{b},{j}},{{Vt,2},{j,d},{l,i}},{{bs,1},{a},{i}},{{Vs,2},{d,b},{a,l}}}
	U11 = {{S,1},{{\[Delta],1},{a},{b}},{{bt,1},{b},{j}},{{Vt,2},{j,d},{l,m}},{{bs,1},{a},{i}},{{Vs,2},{i,d},{l,m}}}
	U12 = {{S,-1},{{bt,1},{b},{j}},{{Vt,2},{j,a},{l,m}},{{bs,1},{a},{i}},{{Vs,2},{i,b},{l,m}}}
	U13 = {{S,-1},{{bt,1},{b},{j}},{{Vt,2},{j,e},{i,m}},{{bs,1},{a},{i}},{{Vs,2},{b,e},{a,m}}}
	U14 = {{S,1},{{bt,1},{b},{j}},{{Vt,2},{j,e},{l,i}},{{bs,1},{a},{i}},{{Vs,2},{b,e},{a,l}}}
	U15 = U12
	U16 = {{S,1},{{\[Delta],1},{a},{b}},{{bt,1},{b},{j}},{{Vt,2},{j,e},{l,m}},{{bs,1},{a},{i}},{{Vs,2},{i,e},{l,m}}}
*/
        
        for (int i=0; i<tno; ++i)
            for (int j=0; j<tno; ++j)
                for (int a=0; a<tnv; ++a)
                    for (int b=0; b<tnv; ++b)
                    {
                        for (int d=0; d<tnv; ++d)
                        {
                            for (int m=0; m<tno; ++m)
                            {
                                e3 -= 0.25*real(bai(a,i)*bai(b,j)*(Vi[i*tn3+(d+tno)*tn2+j*tn+m]-Vi[i*tn3+(d+tno)*tn2+m*tn+j])*(Vi[(d+tno)*tn3+(a+tno)*tn2+(b+tno)*tn+m]-Vi[(d+tno)*tn3+(a+tno)*tn2+m*tn+(b+tno)])/(eigs((d%nv)+no)+eigs((a%nv)+no)-eigs((j%no))-eigs((m%no)) - ee/27.2113));
                                
                                e9 -= 0.25*real(bai(a,i)*bai(b,j)*(Vi[(d+tno)*tn3+(b+tno)*tn2+(a+tno)*tn+m]-Vi[(d+tno)*tn3+(b+tno)*tn2+m*tn+(a+tno)])*(Vi[j*tn3+(d+tno)*tn2+i*tn+m]-Vi[j*tn3+(d+tno)*tn2+m*tn+i])/(eigs((b%nv)+no)+eigs((d%nv)+no)-eigs((i%no))-eigs((m%no)) - ee/27.2113));
                            }
                            
                            for (int e=0; e<tnv; ++e)
                            {
                                e2 -= 0.25*(-1)*real(bai(a,i)*bai(b,j)*(Vi[(d+tno)*tn3+(e+tno)*tn2+(a+tno)*tn+j]-Vi[(d+tno)*tn3+(e+tno)*tn2+j*tn+(a+tno)])*(Vi[(d+tno)*tn3+(e+tno)*tn2+(b+tno)*tn+i]-Vi[(d+tno)*tn3+(e+tno)*tn2+i*tn+(b+tno)])/(eigs((d%nv)+no)+eigs((e%nv)+no)-eigs((j%no))-eigs((i%no)) - ee/27.2113));
                                
                                for (int l=0; l<tno; ++l)
                                {
                                    e6 -= 0.25*real(bai(a,i)*bai(b,j)*(Vi[(d+tno)*tn3+(e+tno)*tn2+(a+tno)*tn+l]-Vi[(d+tno)*tn3+(e+tno)*tn2+l*tn+(a+tno)])*(Vi[(d+tno)*tn3+(e+tno)*tn2+(b+tno)*tn+l]-Vi[(d+tno)*tn3+(e+tno)*tn2+l*tn+(b+tno)])/(eigs((d%nv)+no)+eigs((e%nv)+no)-eigs((l%no))-eigs((i%no)) - ee/27.2113)*delta(i,j));
                                }
                                
                                for (int m=0; m<tno; ++m)
                                {
                                    e1 -= 0.25*real(bai(a,i)*bai(b,j)*(Vi[(d+tno)*tn3+(e+tno)*tn2+(a+tno)*tn+m]-Vi[(d+tno)*tn3+(e+tno)*tn2+m*tn+(a+tno)])*(Vi[(d+tno)*tn3+(e+tno)*tn2+(b+tno)*tn+m]-Vi[(d+tno)*tn3+(e+tno)*tn2+m*tn+(b+tno)])/(eigs((d%nv)+no)+eigs((e%nv)+no)-eigs((j%no))-eigs((m%no)) - ee/27.2113)*delta(i,j));

                                }
                            }
                            for (int l=0; l<tno; ++l)
                            {
                                e7 -= 0.25*(-1)*real(bai(a,i)*bai(b,j)*(Vi[i*tn3+(d+tno)*tn2+l*tn+j]-Vi[i*tn3+(d+tno)*tn2+j*tn+l])*(Vi[(d+tno)*tn3+(a+tno)*tn2+(b+tno)*tn+l]-Vi[(d+tno)*tn3+(a+tno)*tn2+l*tn+(b+tno)])/(eigs((d%nv)+no)+eigs((a%nv)+no)-eigs((l%no))-eigs((j%no)) - ee/27.2113));

                                e10 -= 0.25*(-1)*real(bai(a,i)*bai(b,j)*(Vi[(d+tno)*tn3+(b+tno)*tn2+(a+tno)*tn+l]-Vi[(d+tno)*tn3+(b+tno)*tn2+l*tn+(a+tno)])*(Vi[j*tn3+(d+tno)*tn2+l*tn+i]-Vi[j*tn3+(d+tno)*tn2+i*tn+l])/(eigs((d%nv)+no)+eigs((b%nv)+no)-eigs((l%no))-eigs((i%no)) - ee/27.2113));
                                
                                for (int m=0; m<tno; ++m)
                                {
                                    e11 -= 0.25*real(bai(a,i)*bai(b,j)*(Vi[i*tn3+(d+tno)*tn2+l*tn+m]-Vi[i*tn3+(d+tno)*tn2+m*tn+l])*(Vi[j*tn3+(d+tno)*tn2+l*tn+m]-Vi[j*tn3+(d+tno)*tn2+m*tn+l])/(eigs((d%nv)+no)+eigs((a%nv)+no)-eigs((l%no))-eigs((m%no)) - ee/27.2113)*delta(a,b));
                                }
                                
                            }
                        }

                        for (int e=0; e<tnv; ++e)
                        {
                            for (int m=0; m<tno; ++m)
                            {
                                e4 -= 0.25*(-1)*real(bai(a,i)*bai(b,j)*(Vi[i*tn3+(e+tno)*tn2+j*tn+m]-Vi[i*tn3+(e+tno)*tn2+m*tn+j])*(Vi[(a+tno)*tn3+(e+tno)*tn2+(b+tno)*tn+m]-Vi[(a+tno)*tn3+(e+tno)*tn2+m*tn+(b+tno)])/(eigs((a%nv)+no)+eigs((e%nv)+no)-eigs((j%no))-eigs((m%no)) - ee/27.2113));

				e13 -= 0.25*(-1)*real(bai(a,i)*bai(b,j)*(Vi[(b+tno)*tn3+(e+tno)*tn2+(a+tno)*tn+m]-Vi[(b+tno)*tn3+(e+tno)*tn2+m*tn+(a+tno)])*(Vi[j*tn3+(e+tno)*tn2+i*tn+m]-Vi[j*tn3+(e+tno)*tn2+m*tn+i])/(eigs((b%nv)+no)+eigs((e%nv)+no)-eigs((i%no))-eigs((m%no)) - ee/27.2113));

                                for (int l=0; l<tno; ++l)
                                {
				    e16 -= 0.25*real(bai(a,i)*bai(b,j)*(Vi[i*tn3+(e+tno)*tn2+l*tn+m]-Vi[i*tn3+(e+tno)*tn2+m*tn+l])*(Vi[j*tn3+(e+tno)*tn2+l*tn+m]-Vi[j*tn3+(e+tno)*tn2+m*tn+l])/(eigs((a%nv)+no)+eigs((e%nv)+no)-eigs((l%no))-eigs((m%no)) - ee/27.2113)*delta(a,b));
                                }
                            }

                            for (int l=0; l<tno; ++l)
                            {
     				e8 -= 0.25*real(bai(a,i)*bai(b,j)*(Vi[i*tn3+(e+tno)*tn2+l*tn+j]-Vi[i*tn3+(e+tno)*tn2+j*tn+l])*(Vi[(a+tno)*tn3+(e+tno)*tn2+(b+tno)*tn+l]-Vi[(a+tno)*tn3+(e+tno)*tn2+l*tn+(b+tno)])/(eigs((a%nv)+no)+eigs((e%nv)+no)-eigs((l%no))-eigs((j%no)) - ee/27.2113));                           
				e14 -= 0.25*real(bai(a,i)*bai(b,j)*(Vi[(b+tno)*tn3+(e+tno)*tn2+(a+tno)*tn+l]-Vi[(b+tno)*tn3+(e+tno)*tn2+l*tn+(a+tno)])*(Vi[j*tn3+(e+tno)*tn2+l*tn+i]-Vi[j*tn3+(e+tno)*tn2+i*tn+l])/(eigs((b%nv)+no)+eigs((e%nv)+no)-eigs((l%no))-eigs((i%no)) - ee/27.2113));
                            }
                        }

                        for (int m=0; m<tno; ++m)
                        {
                            for (int l=0; l<tno; ++l)
                            {
				e12 -= 0.25*(-1)*real(bai(a,i)*bai(b,j)*(Vi[i*tn3+(b+tno)*tn2+l*tn+m]-Vi[i*tn3+(b+tno)*tn2+m*tn+l])*(Vi[j*tn3+(a+tno)*tn2+l*tn+m]-Vi[j*tn3+(a+tno)*tn2+m*tn+l])/(eigs((a%nv)+no)+eigs((b%nv)+no)-eigs((l%no))-eigs((m%no)) - ee/27.2113));                                

                            }
                        }
                    }

	dE1 = (e1+e2*2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12*2+e13+e14+e16);
        
        //Term2:
        for (int j=0; j<2*no; ++j)
            for (int b=0; b<2*nv; ++b)
                for (int c=0; c<2*nv; ++c)
                    for (int k=0; k<2*no; ++k)
                        for (int a=0; a<tnv; ++a)
                            Rab(a,b) += 0.5*(Vi[(j)*tn3+(k)*tn2+(b+tno)*tn+(c+tno)] - Vi[(j)*tn3+(k)*tn2+(c+tno)*tn+(b+tno)])*T(c*tnv+a,j*tno+k); // taken from John's dE2, hermitian
        
       for (int a=0; a<2*nv; ++a)
            for (int b=0; b<2*nv; ++b)
                for (int i=0; i<tno; ++i)
		    for (int j=0; j<tno; ++j)
                    	dE2 += real(delta(i,j)*bai(a,i)*bai(b,j)*Rab(a,b)); 
        
        //Term3:
        for (int j=0; j<2*no; ++j)
            for (int a=0; a<2*nv; ++a)
                for (int b=0; b<2*nv; ++b)
                    for (int k=0; k<2*no; ++k)
                        for (int i=0; i<tno; ++i)
                            Rij(i,j) += 0.5*(Vi[(j)*tn3+(k)*tn2+(b+tno)*tn+(a+tno)] - Vi[(j)*tn3+(k)*tn2+(a+tno)*tn+(b+tno)])*T(a*tnv+b,i*tno+k); // taken from John's dE3, hermitian
        
	for (int a=0; a<tnv; ++a)
            for (int b=0; b<tnv; ++b)
		for (int c=0; c<tnv; ++c)
                    for (int i=0; i<tno; ++i)
                    	for (int j=0; j<tno; ++j)
			    dE3 += real(delta(c,a)*delta(c,b)*bai(a,i)*bai(b,j)*Rij(i,j));
 
        //Term4:
        for (int i=0; i<tno; ++i)
            for (int j=0; j<tno; ++j)
                for (int a=0; a<tnv; ++a)
                    for (int b=0; b<tnv; ++b)
                        for (int k=0; k<tno; ++k)
                            for (int c=0; c<tnv; ++c)
                                wai1(i*tnv+a,j*tnv+b) += (Vi[j*tn3+k*tn2+(b+tno)*tn+(c+tno)]-Vi[j*tn3+k*tn2+(c+tno)*tn+(b+tno)])*T(a*tnv+c,i*tno+k);// hermitian
        
        for (int i=0; i<tno; ++i)
            for (int j=0; j<tno; ++j)
                for (int a=0; a<tnv; ++a)
                    for (int b=0; b<tnv; ++b)
                        dE4 += real(bai(a,i)*bai(b,j)*wai1(i*tnv+a,j*tnv+b));

        
        cout << "Term1 - Ex State Corr:    " << dE1 << endl;
        cout << "Term2 - OOV:              " << dE2 << endl;
        cout << "Term3 - VVO:              " << dE3 << endl;
        cout << "Term4 - OV:               " << dE4 << endl;
        cout << "*Total CIS(D) correction: " << (dE1+dE2+dE3+dE4) << " au -- " << (dE1+dE2+dE3+dE4)*27.2113 << " eV" << endl;
        cout << "*Mp2:                     " << Emp2 << endl;
        cout << "*Total:                   " << ee+(dE1+dE2+dE3+dE4)*27.2113 << " eV" << endl;
    }
    
    void CISD3(const double ee, cx_vec v)
    {
	BuildSpinOrbitalV2();
        int tno = 2*no;
        int tnv = 2*nv;
        const cx* Vi = IntCur.memptr();
        
        cx_mat T(4.0*nv*nv,4.0*no*no); T.zeros();
        cx_mat Uabab(nv*nv,no*no); Uabab.zeros();
        cx_mat Uabba(nv*nv,no*no); Uabba.zeros();
        cx_mat Uaaaa(nv*nv,no*no); Uaaaa.zeros();
 
        double dE1=0.0;
    	double dE1S=0.0;
	double dE1O=0.0;
    
        for (int i=0; i<2*no; ++i)
            for (int j=0; j<2*no; ++j)
                for (int a=0; a<2*nv; ++a)
                    for (int b=0; b<2*nv; ++b)
                    {
                        T(a*tnv+b,i*tno+j) -= real((Vi[(a+tno)*tn3+(b+tno)*tn2+i*tn+j]-Vi[(a+tno)*tn3+(b+tno)*tn2+j*tn+i])/(eigs((a%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((j%no))));
                    }
        
        // Check the MP2 energy
        double TestMp2=0.0;
        for (int i=0; i<2*no; ++i)
            for (int j=0; j<2*no; ++j)
                for (int a=0; a<2*nv; ++a)
                    for (int b=0; b<2*nv; ++b)
                    {
                        TestMp2 += real(0.25*(Vi[(a+tno)*tn3+(b+tno)*tn2+i*tn+j]-Vi[(a+tno)*tn3+(b+tno)*tn2+j*tn+i])*T(a*tnv+b,i*tno+j));
                    }
        cout << "*Mp2:                     " << TestMp2 << endl;
        
        // Uabab:
        double fac = 2.0;
        for (int I=0; I<no; ++I)
            for (int j=0; j<no; ++j)
                for (int A=0; A<nv; ++A)
		{
                    for (int b=0; b<nv; ++b)
		    {
                        for (int C=0; C<nv; ++C)
                        {
                            Uabab(A*nv+b,I*no+j) += fac*v(C*no+I)*(Vi[(A+no)*n3+(b+no)*n2+(C+no)*n+j]-Vi[(A+no)*n3+(b+no)*n2+j*n+(C+no)]);
                        }
                        for (int c=0; c<nv; ++c)
                        {
                            Uabab(A*nv+b,I*no+j) += fac*v(c*no+j)*(Vi[(A+no)*n3+(b+no)*n2+I*n+(c+no)]-Vi[(A+no)*n3+(b+no)*n2+(c+no)*n+I]);
                        }
        
                        for (int k=0; k<no; ++k)
                        {
                            Uabab(A*nv+b,I*no+j) += -1.*fac*v(b*no+k)*(Vi[(A+no)*n3+(k)*n2+I*n+j]-Vi[(A+no)*n3+(k)*n2+j*n+I]);
                        }
                        for (int K=0; K<no; ++K)
                        {
                            Uabab(A*nv+b,I*no+j) += -1.*fac*v(A*no+K)*(Vi[K*n3+(b+no)*n2+I*n+j]-Vi[K*n3+(b+no)*n2+j*n+I]);
                        }
		    }
        	}

        // Uabba:
        for (int i=0; i<no; ++i)
            for (int J=0; J<no; ++J)
                for (int A=0; A<nv; ++A)
                    for (int b=0; b<nv; ++b)
		    {
                        for (int c=0; c<nv; ++c)
                        {
                            Uabba(A*nv+b,i*no+J) += -1.*fac*v(c*no+i)*(Vi[(A+no)*n3+(b+no)*n2+J*n+(c+no)]-Vi[(A+no)*n3+(b+no)*n2+(c+no)*n+J]);
			}
                        for (int C=0; C<nv; ++C)
                        {
                            Uabba(A*nv+b,i*no+J) += -1.*fac*v(C*no+J)*(Vi[(A+no)*n3+(b+no)*n2+(C+no)*n+i]-Vi[(A+no)*n3+(b+no)*n2+i*n+(C+no)]);
                        }
        
                        for (int k=0; k<no; ++k)
                        {
                            Uabba(A*nv+b,i*no+J) += fac*v(b*no+k)*(Vi[(A+no)*n3+k*n2+J*n+i]-Vi[(A+no)*n3+k*n2+i*n+J]);
                        }
                        for (int K=0; K<no; ++K)
                        {
                            Uabba(A*nv+b,i*no+J) += fac*v(A*no+K)*(Vi[(K)*n3+(b+no)*n2+J*n+i]-Vi[(K)*n3+(b+no)*n2+i*n+J]);
                        }
        	    } 
        
        // Uaaaa:
        for (int I=0; I<no; ++I)
            for (int J=0; J<no; ++J)
                for (int A=0; A<nv; ++A)
                    for (int B=0; B<nv; ++B)
		    {
                        for (int C=0; C<nv; ++C)
                        {
                            Uaaaa(A*nv+B,I*no+J) += fac*v(C*no+I)*(Vi[(A+no)*n3+(B+no)*n2+(C+no)*n+J]-Vi[(A+no)*n3+(B+no)*n2+J*n+(C+no)]);
                            Uaaaa(A*nv+B,I*no+J) += -1.*fac*v(C*no+J)*(Vi[(A+no)*n3+(B+no)*n2+(C+no)*n+I]-Vi[(A+no)*n3+(B+no)*n2+I*n+(C+no)]);
                        }
                        for (int K=0; K<no; ++K)
                        {
                            Uaaaa(A*nv+B,I*no+J) += fac*v(B*no+K)*(Vi[(K)*n3+(A+no)*n2+I*n+J]-Vi[(K)*n3+(A+no)*n2+J*n+I]);
                            Uaaaa(A*nv+B,I*no+J) += -1.*fac*v(A*no+K)*(Vi[K*n3+(B+no)*n2+I*n+J]-Vi[K*n3+(B+no)*n2+J*n+I]);
                        }
 		    }
        
        //Same spin:
        //Uaaaa2*Uaaaa:
        for (int I=0; I<no; ++I)
            for (int J=0; J<no; ++J)
                for (int A=0; A<nv; ++A)
                    for (int B=0; B<nv; ++B)
                        for (int P=0; P<nv; ++P)// c' -> p
                            for (int Q=0; Q<no; ++Q)// k' -> q
                            {
                                dE1 -= 0.25*fac*real(delta(I,Q)*v(P*no+Q)*(Vi[(A+no)*n3+(B+no)*n2+(P+no)*n+J]-Vi[(A+no)*n3+(B+no)*n2+J*n+(P+no)])*Uaaaa(A*nv+B,I*no+J)/(eigs((A%nv)+no)+eigs((B%nv)+no)-eigs((I%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= -1.*0.25*fac*real(delta(J,Q)*v(P*no+Q)*(Vi[(A+no)*n3+(B+no)*n2+(P+no)*n+I]-Vi[(A+no)*n3+(B+no)*n2+I*n+(P+no)])*Uaaaa(A*nv+B,I*no+J)/(eigs((A%nv)+no)+eigs((B%nv)+no)-eigs((I%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= 0.25*fac*real(delta(B,P)*v(P*no+Q)*(Vi[Q*n3+(A+no)*n2+I*n+J]-Vi[Q*n3+(A+no)*n2+J*n+I])*Uaaaa(A*nv+B,I*no+J)/(eigs((A%nv)+no)+eigs((B%nv)+no)-eigs((I%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= -1.*0.25*fac*real(delta(A,P)*v(P*no+Q)*(Vi[Q*n3+(B+no)*n2+I*n+J]-Vi[Q*n3+(B+no)*n2+J*n+I])*Uaaaa(A*nv+B,I*no+J)/(eigs((A%nv)+no)+eigs((B%nv)+no)-eigs((I%no))-eigs((J%no)) - ee/27.2113));
                            }
        
        //Uabab2*Uabab:
        for (int I=0; I<no; ++I)
            for (int j=0; j<no; ++j)
                for (int A=0; A<nv; ++A)
                    for (int b=0; b<nv; ++b)
		    {
                        for (int P=0; P<nv; ++P)// c' -> p
                        {
                            for (int Q=0; Q<no; ++Q)// k' -> q
                            {
                                dE1 -= 0.25*fac*real(delta(I,Q)*v(P*no+Q)*(Vi[(A+no)*n3+(b+no)*n2+(P+no)*n+j]-Vi[(A+no)*n3+(b+no)*n2+j*n+(P+no)])*Uabab(A*nv+b,I*no+j)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((I%no))-eigs((j%no)) - ee/27.2113));
                                dE1 -= -1.*0.25*fac*real(delta(A,P)*v(P*no+Q)*(Vi[Q*n3+(b+no)*n2+I*n+j]-Vi[Q*n3+(b+no)*n2+j*n+I])*Uabab(A*nv+b,I*no+j)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((I%no))-eigs((j%no)) - ee/27.2113));
                            }
                        }
                        for (int p=0; p<nv; ++p)// c' -> p
                        {
                            for (int q=0; q<no; ++q)// k' -> q
                            {
                                dE1 -= 0.25*fac*real(delta(j,q)*v(p*no+q)*(Vi[(A+no)*n3+(b+no)*n2+I*n+(p+no)]-Vi[(A+no)*n3+(b+no)*n2+(p+no)*n+I])*Uabab(A*nv+b,I*no+j)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((I%no))-eigs((j%no)) - ee/27.2113));
                                dE1 -= -1.*0.25*fac*real(delta(b,q)*v(p*no+q)*(Vi[(A+no)*n3+q*n2+I*n+j]-Vi[(A+no)*n3+q*n2+j*n+I])*Uabab(A*nv+b,I*no+j)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((I%no))-eigs((j%no)) - ee/27.2113));
                            }
                        }
        	    }

        //Uabba2*Uabba:
        for (int i=0; i<no; ++i)
            for (int J=0; J<no; ++J)
                for (int A=0; A<nv; ++A)
                    for (int b=0; b<nv; ++b)
		    {
                        for (int P=0; P<nv; ++P)// c' -> p
                        {
                            for (int Q=0; Q<no; ++Q)// k' -> q
                            {
                                dE1 -= -1.*0.25*fac*real(delta(J,Q)*v(P*no+Q)*(Vi[(A+no)*n3+(b+no)*n2+(P+no)*n+i]-Vi[(A+no)*n3+(b+no)*n2+i*n+(P+no)])*Uabba(A*nv+b,i*no+J)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= -1.*0.25*fac*real(delta(A,P)*v(P*no+Q)*(Vi[Q*n3+(b+no)*n2+i*n+J]-Vi[Q*n3+(b+no)*n2+J*n+i])*Uabba(A*nv+b,i*no+J)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((J%no)) - ee/27.2113));
                            }
                        }
                        for (int p=0; p<nv; ++p)// c' -> p
                        {
                            for (int q=0; q<no; ++q)// k' -> q
                            {
                                dE1 -= -1.*0.25*fac*real(delta(i,q)*v(p*no+q)*(Vi[(A+no)*n3+(b+no)*n2+J*n+(p+no)]-Vi[(A+no)*n3+(b+no)*n2+(p+no)*n+J])*Uabba(A*nv+b,i*no+J)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= 0.25*fac*real(delta(b,p)*v(p*no+q)*(Vi[(A+no)*n3+q*n2+J*n+i]-Vi[(A+no)*n3+q*n2+i*n+J])*Uabba(A*nv+b,i*no+J)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((J%no)) - ee/27.2113));
                            }
                        }
        	    }
	dE1S += dE1;        

        //Opposite spin:
        //Uaaaa2*Uabba
        for (int I=0; I<no; ++I)
            for (int J=0; J<no; ++J)
                for (int A=0; A<nv; ++A)
                    for (int B=0; B<nv; ++B)
                        for (int P=0; P<nv; ++P)// c' -> p
                            for (int Q=0; Q<no; ++Q)// k' -> q
                            {
                                dE1 -= 0.25*fac*real(delta(I,Q)*v(P*no+Q)*Vi[(A+no)*n3+(B+no)*n2+(P+no)*n+J]*Uabba(A*nv+B,I*no+J)/(eigs((A%nv)+no)+eigs((B%nv)+no)-eigs((I%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= -1.*0.25*fac*real(delta(J,Q)*v(P*no+Q)*Vi[(A+no)*n3+(B+no)*n2+(P+no)*n+I]*Uabba(A*nv+B,I*no+J)/(eigs((A%nv)+no)+eigs((B%nv)+no)-eigs((I%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= 0.25*fac*real(delta(B,P)*v(P*no+Q)*Vi[Q*n3+(A+no)*n2+I*n+J]*Uabba(A*nv+B,I*no+J)/(eigs((A%nv)+no)+eigs((B%nv)+no)-eigs((I%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= -1.*0.25*fac*real(delta(A,P)*v(P*no+Q)*Vi[Q*n3+(B+no)*n2+I*n+J]*Uabba(A*nv+B,I*no+J)/(eigs((A%nv)+no)+eigs((B%nv)+no)-eigs((I%no))-eigs((J%no)) - ee/27.2113));
                                
        //Uaaaa2*Uabab
                                dE1 -= 0.25*fac*real(delta(I,Q)*v(P*no+Q)*Vi[(A+no)*n3+(B+no)*n2+(P+no)*n+J]*Uabab(A*nv+B,I*no+J)/(eigs((A%nv)+no)+eigs((B%nv)+no)-eigs((I%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= -1.*0.25*fac*real(delta(J,Q)*v(P*no+Q)*Vi[(A+no)*n3+(B+no)*n2+(P+no)*n+I]*Uabab(A*nv+B,I*no+J)/(eigs((A%nv)+no)+eigs((B%nv)+no)-eigs((I%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= 0.25*fac*real(delta(B,P)*v(P*no+Q)*Vi[Q*n3+(A+no)*n2+I*n+J]*Uabab(A*nv+B,I*no+J)/(eigs((A%nv)+no)+eigs((B%nv)+no)-eigs((I%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= -1.*0.25*fac*real(delta(A,P)*v(P*no+Q)*Vi[Q*n3+(B+no)*n2+I*n+J]*Uabab(A*nv+B,I*no+J)/(eigs((A%nv)+no)+eigs((B%nv)+no)-eigs((I%no))-eigs((J%no)) - ee/27.2113));
                            }
        
        //Uabab2*Uaaaa
        for (int I=0; I<no; ++I)
            for (int j=0; j<no; ++j)
                for (int A=0; A<nv; ++A)
                    for (int b=0; b<nv; ++b)
		    {
                        for (int P=0; P<nv; ++P)// c' -> p
                        {
                            for (int Q=0; Q<no; ++Q)// k' -> q
                            {
                                dE1 -= fac*real(delta(I,Q)*v(P*no+Q)*Vi[(A+no)*n3+(b+no)*n2+(P+no)*n+j]*Uaaaa(A*nv+b,I*no+j)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((I%no))-eigs((j%no)) - ee/27.2113));
                                dE1 -= -1.*fac*real(delta(A,P)*v(P*no+Q)*Vi[Q*n3+(b+no)*n2+I*n+j]*Uaaaa(A*nv+b,I*no+j)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((I%no))-eigs((j%no)) - ee/27.2113));
        //Uabab2*Uabba
                                dE1 -= fac*real(delta(I,Q)*v(P*no+Q)*Vi[(A+no)*n3+(b+no)*n2+(P+no)*n+j]*Uabba(A*nv+b,I*no+j)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((I%no))-eigs((j%no)) - ee/27.2113));
                                dE1 -= -1.*fac*real(delta(A,P)*v(P*no+Q)*Vi[Q*n3+(b+no)*n2+I*n+j]*Uabba(A*nv+b,I*no+j)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((I%no))-eigs((j%no)) - ee/27.2113));
                                
                            }
                        }
                        for (int p=0; p<nv; ++p)// c' -> p
                        {
                            for (int q=0; q<no; ++q)// k' -> q
                            {
                                dE1 -= fac*real(delta(j,q)*v(p*no+q)*Vi[(A+no)*n3+(b+no)*n2+I*n+(p+no)]*Uaaaa(A*nv+b,I*no+j)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((I%no))-eigs((j%no)) - ee/27.2113));
                                dE1 -= -1.*fac*real(delta(b,q)*v(p*no+q)*Vi[(A+no)*n3+q*n2+I*n+j]*Uaaaa(A*nv+b,I*no+j)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((I%no))-eigs((j%no)) - ee/27.2113));
        //Uabab2*Uabba
                                dE1 -= fac*real(delta(j,q)*v(p*no+q)*Vi[(A+no)*n3+(b+no)*n2+I*n+(p+no)]*Uabba(A*nv+b,I*no+j)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((I%no))-eigs((j%no)) - ee/27.2113));
                                dE1 -= -1.*fac*real(delta(b,q)*v(p*no+q)*Vi[(A+no)*n3+q*n2+I*n+j]*Uabba(A*nv+b,I*no+j)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((I%no))-eigs((j%no)) - ee/27.2113));
                            }
                        }
		    }

        //Uabba2*Uaaaa
        for (int i=0; i<no; ++i)
            for (int J=0; J<no; ++J)
                for (int A=0; A<nv; ++A)
                    for (int b=0; b<nv; ++b)
		    {
                        for (int P=0; P<nv; ++P)// c' -> p
                        {
                            for (int Q=0; Q<no; ++Q)// k' -> q
                            {
                                dE1 -= -1.*fac*real(delta(J,Q)*v(P*no+Q)*Vi[(A+no)*n3+(b+no)*n2+(P+no)*n+i]*Uaaaa(A*nv+b,i*no+J)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= -1.*fac*real(delta(A,P)*v(P*no+Q)*Vi[Q*n3+(b+no)*n2+i*n+J]*Uaaaa(A*nv+b,i*no+J)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((J%no)) - ee/27.2113));
        //Uabba2*Uabab
                                dE1 -= -1.*fac*real(delta(J,Q)*v(P*no+Q)*Vi[(A+no)*n3+(b+no)*n2+(P+no)*n+i]*Uabab(A*nv+b,i*no+J)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= -1.*fac*real(delta(A,P)*v(P*no+Q)*Vi[Q*n3+(b+no)*n2+i*n+J]*Uabab(A*nv+b,i*no+J)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((J%no)) - ee/27.2113));
                            }
                        }
                        for (int p=0; p<nv; ++p)// c' -> p
                        {
                            for (int q=0; q<no; ++q)// k' -> q
                            {
                                dE1 -= -1.*0.25*fac*real(delta(i,q)*v(p*no+q)*Vi[(A+no)*n3+(b+no)*n2+J*n+(p+no)]*Uaaaa(A*nv+b,i*no+J)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= 0.25*fac*real(delta(b,p)*v(p*no+q)*Vi[(A+no)*n3+q*n2+J*n+i]*Uaaaa(A*nv+b,i*no+J));
        //Uabba2*Uabab
                                dE1 -= -1.*0.25*fac*real(delta(i,q)*v(p*no+q)*Vi[(A+no)*n3+(b+no)*n2+J*n+(p+no)]*Uabab(A*nv+b,i*no+J)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((J%no)) - ee/27.2113));
                                dE1 -= 0.25*fac*real(delta(b,p)*v(p*no+q)*Vi[(A+no)*n3+q*n2+J*n+i]*Uabab(A*nv+b,i*no+J)/(eigs((A%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((J%no)) - ee/27.2113));
                            }
                        }
		     }
	dE1O += dE1 - dE1S; 

        cout << "Term1 - Ex State Corr:    " << dE1 << " - Same spin: " << dE1S << " - Opp spin: " << dE1O << endl;
        
        }


    // Finds the Chemical Potential, and then the best effective temperature to obtain pops.
    // Based on the current eigs to obtain as close to pops as possible.
    double EffectiveTemperature(const vec& pops)
    {
        if (n_occ+2 >= n_mo or n_occ-2<=0)
            return 0.0;
        cout << "Finding effective Temperature..."  << endl;
        double EvPerAu=27.2113;
        double Kb = 8.61734315e-5/EvPerAu;
        vec reigs = real(eigs);
        vec spops = sort(pops, "descend");
        double dmu = pow(10.0,-3.0); double mui = reigs(n_occ-2); double muf = reigs(n_occ+2); // Assume the chemical potential is bw h-2,l+2
        double hldiff =reigs(n_occ) - reigs(n_occ-1);
        double dB = 10.0*hldiff;
        double Bi=dB; double Bf=100.0/hldiff;
        
        double BestMu = 0.0;
        double BestTemp = 0.0;
        double BestError = n_occ;
        
        for(double mu=mui; mu<muf; mu+=dmu)
        {
            for(double Beta=Bi; Beta<Bf; Beta+=dB)
            {
                vec p = pow(exp(Beta*(reigs-mu))+1.0,-1.0);
                double Error = accu((spops-p)%(spops-p));
                if (Error != Error)
                    Error = 8.0;
                else if (Error < BestError)
                {
                    BestTemp=1.0/(Beta*Kb);
                    BestMu=mu;
                    BestError = Error;
                }
            }
        }
        double Beta = 1.0/(BestTemp*Kb);
        cout << "Effective Temperature: " << BestTemp << " at mu= " << BestMu << " beta= " << Beta << " Fermi-Deviance " << sqrt(BestError) << endl;
        vec p = pow(exp(Beta*(reigs-BestMu))+1.0,-1.0);
        (p-spops).st().print("Deviation From Thermality.");
        return BestTemp;
    }
    
    // Calculate the MP2 energy and opdm blockwise in the fock eigenbasis to avoid singles.
    // Use this as a density guess for the dynamics.
    // Since HF is probably very far from the desired initial state
    cx_mat MP2(cx_mat& RhoHF)
    {
        if (params["ActiveSpace"])
        {
            cout << "PlaceholderXXYXYX" << endl;
            return RhoHF;
        }
        
        typedef std::complex<double> cx;
        
        int n1=nv;
        int n2=nv*n1;
        int n3=n2*no;
        int n4=no*n3;
        std::vector<cx> Vs(n4),T(n4);
        std::fill(Vs.begin(), Vs.end(), std::complex<double>(0.0,0.0));
        std::fill(T.begin(), T.end(), std::complex<double>(0.0,0.0));
        
        cx_mat d2 = Delta2();
        for(int R=0; R<n_aux; ++R)
        {
            cx_mat B = V.t()*BpqR.slice(R)*V;
            for (int i=0; i<no; ++i)
            {
                for (int j=0; j<no; ++j)
                {
                    for (int a=0; a<nv; ++a)
                    {
                        for (int b=0; b<nv; ++b)
                        {
                            Vs[i*n3+j*n2+a*n1+b] += B(i,a+no)*B(j,b+no);
                        }
                    }
                }
            }
        }
        
        Emp2=0.0;
        for (int i=0; i<no; ++i)
        {
            for (int j=0; j<no; ++j)
            {
                for (int a=0; a<nv; ++a)
                {
                    for (int b=0; b<nv; ++b)
                    {
                        T[i*n3+j*n2+a*n1+b] = (2*Vs[i*n3+j*n2+a*n1+b] - Vs[i*n3+j*n2+b*n1+a])/(d2(a+no,i)+d2(b+no,j));
                        //cout << "ijab2" << i << j << a << b << Vs[i*n3+j*n2+a*n1+b] << " " << real(T[i*n3+j*n2+a*n1+b]*Vs[i*n3+j*n2+a*n1+b]) << endl;
                        Emp2 += real(T[i*n3+j*n2+a*n1+b]*Vs[i*n3+j*n2+a*n1+b]);
                    }
                }
            }
        }
        cx_mat Poo(no,no); Poo.zeros();
        cx_mat Pvv(nv,nv); Pvv.zeros();
        cx_mat Pov(no,no); Pov.zeros();
        cx_mat Pmp2(RhoHF);
        
        if (params["Mp2NOGuess"])
        {
            // Now make the PDMs.
            // I'm taking these from JCC Ishimura, Pulay, and Nagase
            // and CPL 166,3 Frisch, HG, Pople.
            for (int i=0; i<no; ++i)
            {
                for (int j=0; j<no; ++j)
                {
                    for (int k=0; k<no; ++k)
                    {
                        for (int a=0; a<nv; ++a)
                        {
                            for (int b=0; b<nv; ++b)
                            {
                                Poo(i,j) -= 2.0*T[i*n3+k*n2+a*n1+b]*Vs[j*n3+k*n2+a*n1+b]/(d2(a+no,k)+d2(b+no,j));
                            }
                        }
                    }
                }
            }
            
            for (int i=0; i<no; ++i)
            {
                for (int j=0; j<no; ++j)
                {
                    for (int c=0; c<nv; ++c)
                    {
                        for (int a=0; a<nv; ++a)
                        {
                            for (int b=0; b<nv; ++b)
                            {
                                Pvv(a,b) += 2.0*T[i*n3+j*n2+a*n1+c]*Vs[i*n3+j*n2+b*n1+c]/(d2(b+no,i)+d2(c+no,j));
                            }
                        }
                    }
                }
            }
            cout << "Tr(oo) " << trace(Poo) << " Tr(vv)" << trace(Pvv) << endl;
            
            // finally do the lag. pieces.
            // Which require new integrals. (-_-)
            cx_mat Lai(nv,no); Lai.zeros();
            for(int R=0; R<n_aux; ++R)
            {
                cx_mat B = V.t()*BpqR.slice(R)*V;
                for (int i=0; i<no; ++i)
                {
                    for (int j=0; j<no; ++j)
                    {
                        for (int a=0; a<nv; ++a)
                        {
                            for (int b=0; b<nv; ++b)
                            {
                                for (int k=0; k<no; ++k)
                                {
                                    Lai(a,i) += Poo(j,k)*(4.0*B(i,a+no)*B(j,k)-2.0*B(i,k)*B(a+no,j));
                                    Lai(a,i) -= 4.0*T[j*n3+k*n2+a*n1+b]*B(i,j)*B(k,b+no);
                                }
                                for (int c=0; c<nv; ++c)
                                {
                                    Lai(a,i) += Pvv(b,c)*(4.0*B(i,a+no)*B(b+no,c+no)-2.0*B(i,b+no)*B(a+no,c+no));
                                    Lai(a,i) += 4.0*T[i*n3+j*n2+b*n1+c]*B(a+no,b+no)*B(j,c+no);
                                }
                            }
                        }
                    }
                }
            }
            Lai.print("Lai");
            // Solve the z-vector equation by iteration.
            // The equation has the form:
            // DP + L + VP = 0
            // Initalize P = L/D
            // Then calculate residual R = -L - VP
            // Then P = -R/D. Iterate.
            double dP=1.0; int iter=100;
            cx_mat dvo=d2.submat(no,0,n-1,no-1);
            cx_mat Pbj=-1.0*Lai/dvo;
            //        dvo.print("dvo");
            //        Pbj.print("Pbj0");
            cx_mat R(Pbj); R.zeros();
            while(dP>pow(10.0,-8.0) && iter>0)
            {
                // Calculate R.
                R = -1.0*Lai;
                for(int S=0; S<n_aux; ++S)
                {
                    cx_mat B = V.t()*BpqR.slice(S)*V;
                    for (int i=0; i<no; ++i)
                    {
                        for (int j=0; j<no; ++j)
                        {
                            for (int a=0; a<nv; ++a)
                            {
                                for (int b=0; b<nv; ++b)
                                {
                                    R(a,i) -= ( 4.0*B(a+no,i)*B(b+no,j)-B(a+no,j)*B(b+no,i)-B(a+no,b+no)*B(i,j) )*Pbj(b,j);
                                }
                            }
                        }
                    }
                }
                dP = accu(abs(Pbj-R/dvo));
                Pbj = R/dvo;
                cout << "Z-Iter:" << 100-iter << " dP: " << dP << endl;
                //            Pbj.print("Pbj");
                iter--;
            }
            Poo.print("Poo");
            Pvv.print("Pvv");
            Pmp2.submat(0,0,no-1,no-1) += Poo;
            Pmp2.submat(no,no,n-1,n-1) += Pvv;
            Pmp2.submat(0,no,no-1,n-1) += Pbj.t();
            Pmp2.submat(no,0,n-1,no-1) += Pbj;
            Pmp2.print("Pmp2");
        }
        return Pmp2;
    }
    
    // Just to see if this is faster than Job 32...
    // Rho is passed in the fock eigenbasis.
    // F is built in the lowdin basis,
    // then transformed into the fock eigenbasis and diagonalized.
    // V_ isn't input. It's output.
    cx_mat FockBuild(cx_mat& Rho, cx_mat& V_, bool debug=false, FieldMatrices* Mus = NULL, double* FS = NULL, int* dir = NULL)
    {
        if (!B_Built)
        {
            cout << "cannot build fock without B" << endl;
            return Rho;
        }
        if (!params["ActiveSpace"])
        {
            mat Xt = X.t();
            cx_mat Vt = V.t();
            F.resize(Rho.n_rows,Rho.n_cols); F.zeros();
            mat J(Rho.n_rows,Rho.n_cols); J.zeros();
            cx_mat K(Rho); K.zeros();
            cx_mat Rhol = V*Rho*Vt; // Get Rho into the lowdin basis for the fock build.
            mat h = 2.0*(Xt)*(H)*X; // H is stored in the ao basis.
            F.set_real(h);
            
#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
            
#pragma omp declare reduction( + : mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< double >( )) ) initializer (omp_priv(omp_orig))
#endif
            
            cx_mat Rhot=Rhol.t();
            
#if HASUDR
#pragma omp parallel for reduction(+:J,K) schedule(static)
#endif
            
            // Fock build should be done in prev eigenbasis instead...
            for (int R=0; R<n_aux; ++R)
            {
                mat B=BpqR.slice(R);
                mat I1=real(Rhol%B); // Because Rho is hermitian and B is hermitian only the real part of this matrix should actually contribute to J
                J += 2.0*B*accu(I1); // J should be a Hermitian matrix.
                K -= B*(Rhot)*B;
            }
            
            F = 0.5*h + J + K;
            if(Mus!=NULL)
            {
                //F in LO basis, i.e. Field must be in LO
                mat FFS(Mus->muxo.n_rows, Mus->muxo.n_cols);
                FFS.zeros();
                FFS.fill((*FS));
                if (*dir == 0)
                    F += FFS % Mus->muxo;
                if (*dir == 1)
                    F += FFS % Mus->muyo;
                if (*dir == 2)
                    F += FFS % Mus->muzo;
                
            }
            
            F = Vt*F*V;    // Move it back into the Prev eigenbasis.
            cx_mat Jx = Vt*J*V;    // Move it back into the Prev eigenbasis.
            cx_mat hx = Vt*h*V;    // Move it back into the Prev eigenbasis.
            K = Vt*K*V;    // Move it back into the Prev eigenbasis.
            
            //            Jx.print("EbJ");
            //            K.print("EbK");
            
            F = 0.5*(F+F.t());
            
            double Eone= real(trace(Rho*hx));
            double EJ= real(trace(Rho*Jx));
            double EK= real(trace(Rho*K));
            Ehf = Eone + EJ + EK + nuclear_energy;
            
            cx_mat Vprime; eigs.zeros();
            if (!eig_sym(eigs, Vprime, F))
            {
                cout << "eig_sym failed to diagonalize fock matrix ... " << endl;
                throw 1;
            }
            
            if (params["Print"]>0.0)
            {
                cout << std::setprecision(9) << endl;
                Rhol.print("Rhol");
                cout << "HF Energy" << Ehf << endl;
                cout << "One Energy" << Eone << endl;
                cout << "J Energy" << EJ << endl;
                cout << "K Energy" << EK << endl;
                cout << "nuclear_energy " << nuclear_energy << endl << endl;
                eigs.st().print("Eigs");
            }
            
            V_ = V = V*Vprime;
            C = X*V;
            
            
            return Vprime;
        }
        else
        {
            // Build in the original Fock basis. to save on B?
            mat Xt = X.t();
            cx_mat Vt = V.t();
            cx_mat Vst = asp->Vs.t();
            F.resize(n_mo,n_mo); F.zeros();
            mat J(n_mo,n_mo); J.zeros();
            cx_mat K(n_mo,n_mo); K.zeros();
            cx_mat Rhol = asp->Vs*Rho*Vst+asp->P0cx; // Get Rho into the lowdin basis for the fock build.
            mat h = 2.0*(Xt)*(H)*X; // h is stored in the ao basis.
            F.set_real(h);
            
#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
            
#pragma omp declare reduction( + : mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< double >( )) ) initializer (omp_priv(omp_orig))
#endif
            
            cx_mat Rhot=Rhol.t();
            
#if HASUDR
#pragma omp parallel for reduction(+:J,K) schedule(static)
#endif
            // Note that I will wastefully make the entire fock matrix for AS. this should be fixed.
            for (int R=0; R<n_aux; ++R)
            {
                mat B=BpqR.slice(R);
                mat I1=real(Rhol%B); // Because Rho is hermitian and B is hermitian only the real part of this matrix should actually contribute to J
                J += 2.0*B*accu(I1); // J should be a Hermitian matrix.
                K -= B*(Rhot)*B;
            }
            
            F = 0.5*h + J + K;
            cx_mat Fs = Vst*F*(asp->Vs);    // Move it back into the Prev eigenbasis.
            F = 0.5*(F+F.t()); // F now has dim Nact X Nact
            
            double Eone= real(trace(Rhol*h));
            double EJ= real(trace(Rhol*J));
            double EK= real(trace(Rhol*K));
            Ehf = Eone + EJ + EK + nuclear_energy;
            
            cx_mat Vprime; eigs.zeros();
            if (!eig_sym(eigs, Vprime, Fs))
            {
                cout << "eig_sym failed to diagonalize fock matrix ... " << endl;
                throw 1;
            }
            
            if (params["Print"]>0.0)
            {
                cout << std::setprecision(9) << endl;
                cout << "HF Energy" << Ehf << endl;
                cout << "One Energy" << Eone << endl;
                cout << "J Energy" << EJ << endl;
                cout << "K Energy" << EK << endl;
                cout << "nuclear_energy " << nuclear_energy << endl << endl;
                eigs.st().print("Eigs");
            }
            
            cx_mat Vprimeb(n_mo,n_mo); Vprimeb.eye(); // Big basis's rotation.
            Vprimeb.submat(asp->ActiveFirst,asp->ActiveFirst,asp->ActiveLast,asp->ActiveLast) = Vprime;
            
            V_ = V = V*Vprimeb;
            asp->Vs = asp->Vs*Vprime;
            
            C = X*V;
            asp->Csub = X*asp->Vs;
            return Vprime;
        }
    }
    
    cx_mat Delta2(bool Sign=true)
    {
        if (eigs.n_elem == 0)
            cout << "Err.." << endl;
        cx_mat tore(eigs.n_elem,eigs.n_elem);
        for(int i=0; i<eigs.n_elem; ++i)
            for(int j=0; j<eigs.n_elem; ++j)
                tore(i,j) = (Sign)? (eigs(i) - eigs(j)) : eigs(i) + eigs(j);
        tore.diag().zeros();
        return tore;
    }
    
    // Updates BpqR
    void BuildB()
    {
        cout << "Building B..." << endl;
        if (!params["ActiveSpace"])//(!params["ActiveSpace"])
        {
            FileMan(FM_WRITE, FILE_MO_COEFS, FM_DP, n_mo*n_mo, 0, FM_BEG, X.memptr(), 0); // B is now built in the Lowdin basis and kept that way.
            BpqR*=0.0;
            mat Btmp(n_mo,n_aux);
            INTEGER ip[2], iq[2], iFile;
            iq[0] = 1; iq[1] = n_mo;
            ip[0] = 1; ip[1] = n_mo;
            int p_len = ip[1]-ip[0]+1;
            int q_len = iq[1]-iq[0]+1;
            iFile = FILE_3C2Ea_INTS;
            threading_policy::enable_omp_only();
            
            double* qas=qalloc_start();
            int LenV = megtot();
            formpqr(&iFile, iq, ip, qas, &LenV);
            threading_policy::pop();
            cout << "FormedPQR..." << endl;
            // Effn' qchem and its BS disk communication...
            // ($*%(*$%*@(@(*
            int fileNum = FILE_BIJQa_INTS;
            for (int q=0; q<q_len; q++)
            {
                int Off8 = q * p_len * n_aux; Btmp *= 0.0;
                LongFileMan(FM_READ, FILE_3C2Ea_INTS, FM_DP, p_len*n_aux, Off8, FM_BEG, Btmp.memptr(), 0);
                // Q.subcube( first_row, first_col, first_slice, last_row, last_col, last_slice )
                BpqR.subcube(q,0,0,q,n_mo-1,n_aux-1) += Btmp*RSinv;
            }
            // Ensure B is perfectly Hermitian.
            for (int R=0; R<n_aux; ++R)
                BpqR.slice(R) = 0.5*(BpqR.slice(R)+BpqR.slice(R).t());
            if (false)
            {
                // This works! Keeping it in as a check for now.
                double* f_diag = new double[n_mo];
                FileMan(FM_READ, FILE_MO_COEFS, FM_DP, n_mo, n_mo*n_mo*2 , FM_BEG, f_diag, 0);
                std::complex<double> ECorr = 0.0;
                cx_mat Vt = V.t();
                for (int R=0; R<n_aux; ++R)
                {
                    cx_mat bR = Vt*BpqR.slice(R)*V;
                    for (int S=0; S<n_aux; ++S)
                    {
                        cx_mat bS = Vt*BpqR.slice(S)*V;
                        for (int i=0; i<n_occ; ++i)
                            for (int j=0; j<n_occ; ++j)
                                for (int a=n_occ; a<n_mo; ++a)
                                    for (int b=n_occ; b<n_mo; ++b)
                                    {
                                        double d = (f_diag[a]+f_diag[b]-f_diag[i]-f_diag[j]);
                                        {
                                            ECorr += bR(i,a)*bR(j,b)*(2.0*bS(i,a)*bS(j,b) - bS(i,b)*bS(j,a))/d;
                                        }
                                    }
                    }
                }
                cout << "Mp2 Correlation Energy: " << std::setprecision(9) << ECorr << endl;
            }
            B_Built = true;
            cout << "BpqR built ..." << endl;
        }
        else
        {
            FileMan(FM_WRITE, FILE_MO_COEFS, FM_DP, n_mo*n_mo, 0, FM_BEG, X.memptr(), 0); // B is now built in the Lowdin basis and kept that way.
            BpqR*=0.0;
            mat Btmp(n_mo,n_aux);
            INTEGER ip[2], iq[2], iFile;
            iq[0] = 1+asp->ActiveFirst; iq[1] = 1+asp->ActiveLast;
            ip[0] = 1+asp->ActiveFirst; ip[1] = 1+asp->ActiveLast;
            int p_len = ip[1]-ip[0]+1;
            int q_len = iq[1]-iq[0]+1;
            iFile = FILE_3C2Ea_INTS;
            threading_policy::enable_omp_only();
            
            double* qas=qalloc_start();
            int LenV = megtot();
            formpqr(&iFile, iq, ip, qas, &LenV);
            threading_policy::pop();
            cout << "FormedPQR..." << endl;
            // Effn' qchem and its BS disk communication...
            // ($*%(*$%*@(@(*
            int fileNum = FILE_BIJQa_INTS;
            for (int q=0; q<q_len; q++)
            {
                int Off8 = q * p_len * n_aux; Btmp *= 0.0;
                LongFileMan(FM_READ, FILE_3C2Ea_INTS, FM_DP, p_len*n_aux, Off8, FM_BEG, Btmp.memptr(), 0);
                // Q.subcube( first_row, first_col, first_slice, last_row, last_col, last_slice )
                BpqR.subcube(q,0,0,q,n_mo-1,n_aux-1) += Btmp*RSinv;
            }
            // Ensure B is perfectly Hermitian.
            for (int R=0; R<n_aux; ++R)
                BpqR.slice(R) = 0.5*(BpqR.slice(R)+BpqR.slice(R).t());
            B_Built = true;
            cout << "BpqR built ..." << endl;
        }
    }
    
    bool cx_mat_equals(const arma::cx_mat& X,const arma::cx_mat& Y, double tol=pow(10.0,-12.0)) const
    {
        if (X.n_rows != Y.n_rows)
            return false;
        if (X.n_cols != Y.n_cols)
            return false;
        bool close(false);
        if(arma::max(arma::max(arma::abs(X-Y))) < tol)
            close = true;
        return close;
    }
    
    void UpdateVi(const cx_mat& V_)
    {
        if (params["Print"])
            cout << "UpdateVi" << endl;
        if (cx_mat_equals(V_,VCur))
            return;
        cx_mat Vi(IntCur.memptr(),n2,n2,false,true); Vi.zeros(); // Vi literally uses the memory of IntCur. This gets around OMP spec issues.
        
        if (InCore)
        {
#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
            
#pragma omp parallel for reduction(+: Vi) schedule(static)
#endif
            for(int R=0; R<n_aux; ++R)
            {
                cx_mat B;
                if (params["ActiveSpace"])
                    B = (asp->Vs).t()*BpqR.slice(R)*(asp->Vs); // No need to xform 5 times/iteration. I could do this once for a 5x speedup.
                else
                    B = V_.t()*BpqR.slice(R)*V_; // No need to xform 5 times/iteration. I could do this once for a 5x speedup.
                Vi += kron(B,B).st();
                B.clear();
            }
        }
        else
        {
            for(int R=0; R<n_aux; ++R)
            {
                cx_mat B;
                if (params["ActiveSpace"])
                    B = (asp->Vs).t()*BpqR.slice(R)*(asp->Vs); // No need to xform 5 times/iteration. I could do this once for a 5x speedup.
                else
                    B = V_.t()*BpqR.slice(R)*V_; // No need to xform 5 times/iteration. I could do this once for a 5x speedup.
                // Manually OMP the kron to avoid memory death.
#pragma omp parallel for schedule(guided)
                for(int p=0; p<n; ++p)
                {
                    for(int q=0; q<n; ++q)
                    {
                        cx A = B(p,q);
                        if (abs(A) < thresh)
                            continue;
                        for(int r=0; r<n; ++r)
                        {
                            for(int s=0; s<n; ++s)
                            {
                                Vi(p*n+r,q*n+s) += A*B(r,s);
                            }
                        }
                    }
                }
                B.clear();
            }
        }
        
         
         // Check the integrals.
         cx_mat d2 = Delta2();
         Emp2=0.0;
         for (int i=0; i<no; ++i)
         {
         for (int j=0; j<no; ++j)
         {
         for (int a=no; a<n; ++a)
         {
         for (int b=no; b<n; ++b)
         {
         //cout << "ijab" << i << j << a << b << IntCur[i*n3+j*n2+a*n+b] << "::" << real(IntCur[i*n3+j*n2+a*n+b]*(2*IntCur[i*n3+j*n2+a*n+b] - IntCur[i*n3+j*n2+b*n+a])/(d2(a,i)+d2(b,j))) << endl;
         Emp2 += real(IntCur[i*n3+j*n2+a*n+b]*(2*IntCur[i*n3+j*n2+a*n+b] - IntCur[i*n3+j*n2+b*n+a])/(d2(a,i)+d2(b,j)));
         }
         }
         }
         }
         cout << std::setprecision(9) << "Mp2 Correlation energy: " << Emp2 << endl;
         
        VCur = V_;
        if (params["Print"])
            cout << "~UpdateVi" << endl;
    }
    
    void Split_RK4_Step_MMUT(const arma::vec& eigval, const arma::cx_mat& Cu , const arma::cx_mat& oldrho, arma::cx_mat& newrho, const double tnow, const double dt)
    {
        cx_mat Ud = exp(eigval*-0.5*j*dt);
        cx_mat U = Cu*diagmat(Ud)*Cu.t();
        cx_mat RhoHalfStepped = U*oldrho*U.t();
        
        arma::cx_mat k1(newrho),k2(newrho),k3(newrho),k4(newrho),v2(newrho),v3(newrho),v4(newrho);
        k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
        v2.zeros(); v3.zeros(); v4.zeros();
        
        RhoDot( RhoHalfStepped, k1,tnow);
        v2 = (dt/2.0) * k1;
        v2 += RhoHalfStepped;
        RhoDot(  v2, k2,tnow+(dt/2.0));
        v3 = (dt/2.0) * k2;
        v3 += RhoHalfStepped;
        RhoDot(  v3, k3,tnow+(dt/2.0));
        v4 = (dt) * k3;
        v4 += RhoHalfStepped;
        RhoDot(  v4, k4,tnow+dt);
        newrho = RhoHalfStepped;
        newrho += dt*(1.0/6.0)*k1;
        newrho += dt*(2.0/6.0)*k2;
        newrho += dt*(2.0/6.0)*k3;
        newrho += dt*(1.0/6.0)*k4;
        newrho = U*newrho*U.t();
    }
    
    
    // Two mmut steps of length dt/2 and one ee2 step (calc'd with RK4)
    double step(rhf* the_scf, FieldMatrices* Mus, arma::vec& eigs_, arma::cx_mat& V_,  arma::cx_mat& Rho_, arma::cx_mat& RhoM12_, const double tnow , const double dt,bool& IsOn)
    {
#ifdef _OPENMP
        if (omp_get_max_threads() <=1)
            omp_set_num_threads(8);
#endif
        if (params["TDTDA"])
        {
            return TDTDAstep( the_scf,  Mus, eigs_, V_, Rho_, RhoM12_, tnow , dt, IsOn);
        }
        else if (params["BBGKY"])
        {
            return BBGKYstep( the_scf,  Mus, eigs_, V_, Rho_, RhoM12_, tnow , dt, IsOn);
        }
        else if (params["StaticStep"])
        {
            // No constant transformation based off what I learned about stability of BBGKY
            // MP partitioning is fixed at T=0
            return StaticStep( the_scf,  Mus, eigs_, V_, Rho_, RhoM12_, tnow , dt, IsOn);
        }
        
        
        if (params["Print"])
            cout << "step" << endl;
        cx_mat newrho(Rho_);
        
        cx_mat Rot;
        if (!params["TDCIS"])
            Rot = FockBuild(Rho_, V_); // updates eigs, C, V, F, etc. etc. etc.
        else
        {
            Rot.resize(n,n);
            Rot.eye();
        }
        
        if (!(params["RIFock"] || params["TDCIS"] || params["TDTDA"]|| params["Corr"]))
            UpdateVi(V_); // Updates CurrInts to avoid rebuilds in the steps that follow.
        
        vec noc0; cx_mat nos0;
        if (params["Print"]>0.0)
            eig_sym(noc0,nos0,Rho_);
        
        
        if (!params["TDCIS"])
        {
            eigs_ = eigs;
            if (!params["ActiveSpace"])
                Mus->update(X*V_);
            else
                Mus->update(X*asp->Csub);
            Rho_ = Rot.t()*Rho_*Rot;
            RhoM12_ = Rot.t()*RhoM12_*Rot;
        }
        
        // Make the exponential propagator.
        arma::cx_mat F(Rho_); F.zeros(); F.set_real(diagmat(eigs));
        if (Mus!=NULL)
            Mus->ApplyField(F,tnow,IsOn);
        vec Gd; cx_mat Cu;
        eig_sym(Gd,Cu,F);
        
        // Full step RhoM12 to make new RhoM12.
        cx_mat NewRhoM12(Rho_);
        cx_mat NewRho(Rho_);
        Split_RK4_Step_MMUT(Gd, Cu, RhoM12_, NewRhoM12, tnow, dt);
        
        // Half step that to make the new Rho.
        Split_RK4_Step_MMUT(Gd, Cu, NewRhoM12, NewRho, tnow, dt/2.0);
        Rho_ = 0.5*(NewRho+NewRho.t());//0.5*(NewRho+NewRho.t());
        RhoM12_ = 0.5*(NewRhoM12+NewRhoM12.t());//0.5*(NewRhoM12+NewRhoM12.t());
        
        //Epseudo=real(RhoDot(Rho_, NewRho, tnow));
        
        // Get the change in noccs.
        if (params["Print"]>1.0)
        {
            vec noc1; cx_mat nos1;
            eig_sym(noc1,nos1,Rho_);
            noc1-=noc0;
            noc1.st().print("Delta N-Occ Nums: ");
        }
        if (params["Print"])
            cout << "~step" << endl;
        
        return Ehf;
    }
    
    
    double TDTDAstep(rhf* the_scf, FieldMatrices* Mus, arma::vec& eigs_, arma::cx_mat& V_,  arma::cx_mat& Rho_, arma::cx_mat& RhoM12_, const double tnow , const double dt,bool& IsOn)
    {
        
        mat Xt = X.t();
        cx_mat Vt = V.t();
        cx_mat hmu = Vt*(Xt)*(H)*X*V;
        cx_mat dRho_(Rho_);
        
        BuildSpinOrbitalV();
        
        cx_mat newrho(Rho_); newrho.zeros();
        //First Term
        
        newrho = Rho_;
        //rk4steps
        if (params["SOTDTDA"])
        {
            arma::cx_mat k1(newrho),k2(newrho),k3(newrho),k4(newrho),v2(newrho),v3(newrho),v4(newrho);
            k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
            v2.zeros(); v3.zeros(); v4.zeros();
            
            sotdtda( Rho_, k1,tnow,hmu,IsOn,Mus);
            v2 = (dt/2.0) * k1;
            v2 += Rho_;
            
            sotdtda(  v2, k2,tnow+(dt/2.0),hmu,IsOn,Mus);
            v3 = (dt/2.0) * k2;
            v3 += Rho_;
            
            sotdtda(  v3, k3,tnow+(dt/2.0),hmu,IsOn,Mus);
            v4 = (dt) * k3;
            v4 += Rho_;
            
            sotdtda(  v4, k4,tnow+dt,hmu,IsOn,Mus);
            cx_mat tmp(k1); tmp.zeros();
            tmp = dt*(1.0/6.0)*k1 + dt*(2.0/6.0)*k2 + dt*(2.0/6.0)*k3 + dt*(1.0/6.0)*k4;
            newrho += tmp;
        }
        else
        {
            
            arma::cx_mat k1(newrho),k2(newrho),k3(newrho),k4(newrho),v2(newrho),v3(newrho),v4(newrho);
            k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
            v2.zeros(); v3.zeros(); v4.zeros();
            
            tdtda( Rho_, k1,tnow,hmu,IsOn,Mus);
            v2 = (dt/2.0) * k1;
            v2 += Rho_;
            
            tdtda(  v2, k2,tnow+(dt/2.0),hmu,IsOn,Mus);
            v3 = (dt/2.0) * k2;
            v3 += Rho_;
            
            tdtda(  v3, k3,tnow+(dt/2.0),hmu,IsOn,Mus);
            v4 = (dt) * k3;
            v4 += Rho_;
            
            tdtda(  v4, k4,tnow+dt,hmu,IsOn,Mus);
            cx_mat tmp(k1); tmp.zeros();
            tmp = dt*(1.0/6.0)*k1 + dt*(2.0/6.0)*k2 + dt*(2.0/6.0)*k3 + dt*(1.0/6.0)*k4;
            newrho += tmp;
            
        }
        if (params["Print"])
        {
            dRho_ = newrho - dRho_;
            cout << "Time: " << tnow << endl;
            dRho_.print("dRho");
            (newrho*newrho-newrho).print("rho2-rho");
            cout << endl << endl;
        }
        Rho_ = newrho;
        RhoM12_=newrho;
        
        return MeanFieldEnergy(Rho_);
        
    }
    // The updates are done in spin-orbital form. and l2 is stored in spin-orbital form.
    // However rho is returned spin-adapted. It's the fewest lines of code way to get that done.
    double BBGKYstep(rhf* the_scf, FieldMatrices* Mus, arma::vec& eigs_, arma::cx_mat& V_,  arma::cx_mat& Rho_, arma::cx_mat& RhoM12_, const double tnow , const double dt,bool& IsOn)
    {
        if (params["Print"])
            cout << "BBGKY step" << endl;
        
        // Make the exponential propagator for fock and dipole parts.
        mat Xt = X.t();
        cx_mat Vt = V.t();
        cx_mat hmu = Vt*(Xt)*(H)*X*V;
        cx_mat h = hmu;
        
        if(Mus!=NULL)
            Mus->ApplyField(hmu,tnow,IsOn);
        vec Gd; cx_mat Cu;
        eig_sym(Gd,Cu,hmu);
        cx_mat Ud = exp(Gd*-0.5*j*dt);
        cx_mat U = Cu*diagmat(Ud)*Cu.t();
        cx_mat RhoHalfStepped = U*Rho_*U.t();
        
        BuildSpinOrbitalV();
        
        cx_mat newrho(Rho_); newrho.zeros();
        // one body step.
        {
            arma::cx_mat k1(newrho),k2(newrho),k3(newrho),k4(newrho),v2(newrho),v3(newrho),v4(newrho);
            k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
            v2.zeros(); v3.zeros(); v4.zeros();
            
            bbgky1( RhoHalfStepped, k1,tnow);
            v2 = (dt/2.0) * k1;
            v2 += RhoHalfStepped;
            
            bbgky1(  v2, k2,tnow+(dt/2.0));
            v3 = (dt/2.0) * k2;
            v3 += RhoHalfStepped;
            
            bbgky1(  v3, k3,tnow+(dt/2.0));
            v4 = (dt) * k3;
            v4 += RhoHalfStepped;
            
            bbgky1(  v4, k4,tnow+dt);
            newrho = RhoHalfStepped;
            newrho += dt*(1.0/6.0)*k1;
            newrho += dt*(2.0/6.0)*k2;
            newrho += dt*(2.0/6.0)*k3;
            newrho += dt*(1.0/6.0)*k4;
        }
        // Two-body step.
        if (!params["BBGKY1"])
        {
            Transform2(r2,U);
            
            arma::cx_mat k1(r2),k2(r2),k3(r2),k4(r2),v2(r2),v3(r2),v4(r2);
            k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
            v2.zeros(); v3.zeros(); v4.zeros();
            
            bbgky2( r2, k1,tnow);
            v2 = (dt/2.0) * k1;
            v2 += r2;
            bbgky2(  v2, k2,tnow+(dt/2.0));
            v3 = (dt/2.0) * k2;
            v3 += r2;
            bbgky2(  v3, k3,tnow+(dt/2.0));
            v4 = (dt) * k3;
            v4 += r2;
            bbgky2(  v4, k4,tnow+dt);
            r2 += dt*(1.0/6.0)*k1;
            r2 += dt*(2.0/6.0)*k2;
            r2 += dt*(2.0/6.0)*k3;
            r2 += dt*(1.0/6.0)*k4;
            
            Transform2(r2,U);
        }
        
        Rho_ = U*newrho*U.t();
        RhoM12_ = U*newrho*U.t();
        //   TracedownConsistency(Rho_,r2); // Ensures the separable part = Rho at all times.
        
        if (!(params["BBGKY1"]) && !(params["TDTDA"]))
            return CorrelatedEnergy(r2);
        else
        {
            return MeanFieldEnergy(Rho_);
        }
    }
    
    // Do the eigenvalue build once and then fix them.
    // That's basically the Tamm-dancoff approximation.
    double StaticStep(rhf* the_scf, FieldMatrices* Mus, arma::vec& eigs_, arma::cx_mat& V_,  arma::cx_mat& Rho_, arma::cx_mat& RhoM12_, const double tnow , const double dt,bool& IsOn)
    {
        // Make the exponential propagator for fock and dipole parts.
        mat Xt = X.t();
        cx_mat Vt = V.t();
        mat h = (Xt)*(H)*X;
        cx_mat hmu = Vt*h*V;
        if(Mus!=NULL)
            Mus->ApplyField(hmu,tnow,IsOn);
        FockBuildNoRotation(Rho_, hmu, tnow);
        vec Gd; cx_mat Cu;
        eig_sym(Gd,Cu,hmu);
        cx_mat Ud = exp(Gd*-0.5*j*dt);
        cx_mat U = Cu*diagmat(Ud)*Cu.t();
        cx_mat RhoHalfStepped = U*Rho_*U.t();
        
        cx_mat newrho(Rho_); newrho.zeros();
        // one body step.
        {
            arma::cx_mat k1(newrho),k2(newrho),k3(newrho),k4(newrho),v2(newrho),v3(newrho),v4(newrho);
            k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
            v2.zeros(); v3.zeros(); v4.zeros();
            
            RhoDot( RhoHalfStepped, k1,tnow);
            v2 = (dt/2.0) * k1;
            v2 += RhoHalfStepped;
            
            RhoDot(  v2, k2,tnow+(dt/2.0));
            v3 = (dt/2.0) * k2;
            v3 += RhoHalfStepped;
            
            RhoDot(  v3, k3,tnow+(dt/2.0));
            v4 = (dt) * k3;
            v4 += RhoHalfStepped;
            
            RhoDot(  v4, k4,tnow+dt);
            newrho = RhoHalfStepped;
            newrho += dt*(1.0/6.0)*k1;
            newrho += dt*(2.0/6.0)*k2;
            newrho += dt*(2.0/6.0)*k3;
            newrho += dt*(1.0/6.0)*k4;
        }
        
        Rho_ = U*newrho*U.t();
        RhoM12_ = U*newrho*U.t();
        
        return Ehf;
    }
    
    void BuildSpinOrbitalV()
    {
        cx* Vi = IntCur.memptr();
        
        // Alpha (0,n) Beta (n,2n) Ordering.
        Vso.resize(4*n2,4*n2);
        Vso.zeros();
        cx* vso = Vso.memptr();
        int s1,s2,s3,s4;
        int P,Q,R,S;
        
        for (int p=0;p<tn;++p)
            for (int q=0;q<tn;++q)
                for (int r=0;r<tn;++r)
                    for (int s=0;s<tn;++s)
                    {
                        s1=p<n;s2=q<n;s3=r<n;s4=s<n;
                        P=p%n; Q=q%n; R=r%n; S=s%n;
                        if ((s1!=s3) || (s2!=s4))
                            continue;
                        vso[tn3*p+tn2*q+tn*r+s] = Vi[P*n3+Q*n2+R*n+S];
                    }
        
    }
    
    int SpinOrbitalOffset(int i)
    {
        // The input is mapped a 0 --- a n | beta 0 --- beta n
        // the output is all a -- no b -- no | etc.
        if (i<no)
            return i;
        if (i>=no && i<n) //alpha virt.
            return (i-no)+2*no;
        if (i>=n && i<(n+no)) // beta occ
            return (i-n)+no;
        else
            return (i-n-no)+(2*no+nv); // beta virt
        
    }
    
    void BuildSpinOrbitalV2()
    {
        cx* Vi = IntCur.memptr();
        // Alpha (0,no) Beta (0,no), Alpha (no,n) Beta (no,n) Ordering.
        Vso.resize(4*n2,4*n2);
        Vso.zeros();
        cx* vso = Vso.memptr();
        int s1,s2,s3,s4;
        int P,Q,R,S;
        int P2,Q2,R2,S2;
        
        /*
        cx_mat t(n2,n2); t.zeros();
        cx* T = t.memptr();
        Emp2=0.0;
        for (int i=0; i<no; ++i)
        {
            for (int j=0; j<no; ++j)
            {
                for (int a=no; a<n; ++a)
                {
                    for (int b=no; b<n; ++b)
                    {
                        cx d = eigs(a)-eigs(i)+eigs(b)-eigs(j); //(eigs(a)*eta(a)+eigs(b)*eta(b)-eigs(i)*rho(i)-eigs(j)*rho(j));
                        if (abs(d) > 0.0000001)
                        {
                            T[a*n3+b*n2+i*n+j] = (2*Vi[i*n3+j*n2+a*n+b] - Vi[i*n3+j*n2+b*n+a])/d;
                            Emp2 += real(T[a*n3+b*n2+i*n+j]*Vi[i*n3+j*n2+a*n+b]);
                        }
                    }
                }
            }
        }
        cout << "Emp2: " << Emp2 << endl; */
        for (int p=0;p<tn;++p)
            for (int q=0;q<tn;++q)
                for (int r=0;r<tn;++r)
                    for (int s=0;s<tn;++s)
                    {
                        s1=p<n;s2=q<n;s3=r<n;s4=s<n;
                        P=p%n; Q=q%n; R=r%n; S=s%n;
                        if ((s1!=s3) || (s2!=s4))
                            continue;
                        P2 = SpinOrbitalOffset(p);
                        Q2 = SpinOrbitalOffset(q);
                        R2 = SpinOrbitalOffset(r);
                        S2 = SpinOrbitalOffset(s);
                        
                        vso[tn3*P2+tn2*Q2+tn*R2+S2] = Vi[P*n3+Q*n2+R*n+S];
                        
//                        if (abs(vso[tn3*P2+tn2*Q2+tn*R2+S2]) < pow(10.0,-9.0))
//                            continue;
//                        cout << s1<< s2 << s3 << s4 << " : " << P2 << " : " << Q2 << " : " << R2 << " : " << S2 << " | " << p << " : " << q << " : " << r << " : " << s << endl;
                        
                    }
        
        /*
        {
            int tnv=nv*2;
            int tno=no*2;
        cx_mat T(4.0*nv*nv,4.0*no*no); T.zeros();
        for (int i=0; i<2*no; ++i)
            for (int j=0; j<2*no; ++j)
                for (int a=0; a<2*nv; ++a)
                    for (int b=0; b<2*nv; ++b)
                    {
                        
                        s1=i<no;s2=j<no;s3=a<(tno+nv);s4=b<(tno+nv);
//                        if ((s1+s2) != (s3+s4))
//                            continue;
                        if (abs(vso[(a+tno)*tn3+(b+tno)*tn2+i*tn+j]) < pow(10.0,-9.0))
                            continue;
                        T(a*tnv+b,i*tno+j) -= real((vso[(a+tno)*tn3+(b+tno)*tn2+i*tn+j]-vso[(a+tno)*tn3+(b+tno)*tn2+j*tn+i])/(eigs((a%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((j%no))));
                        //cout << s1<< s2 << s3 << s4 << " : " << i << " : " << j << " : " << a << " : " << b <<  " T: "<<  T(a*tnv+b,i*tno+j) << " " << vso[(a+tno)*tn3+(b+tno)*tn2+i*tn+j]<< " " << vso[(a+tno)*tn3+(b+tno)*tn2+j*tn+i] << " " << (eigs((a%nv)+no)+eigs((b%nv)+no)-eigs((i%no))-eigs((j%no))) << endl;
                        
                        
                    }

        double TestMp2=0.0;
        for (int i=0; i<2*no; ++i)
            for (int j=0; j<2*no; ++j)
                for (int a=0; a<2*nv; ++a)
                    for (int b=0; b<2*nv; ++b)
                    {
                        s1=i<no;s2=j<no;s3=a<(tno+nv);s4=b<(tno+nv);
                        if ((s1+s2) != (s3+s4))
                            continue;

                        TestMp2 += real(0.25*(vso[(a+tno)*tn3+(b+tno)*tn2+i*tn+j]-vso[(a+tno)*tn3+(b+tno)*tn2+j*tn+i])*T(a*2*nv+b,i*2*no+j));
                        cout << "Test MP2RR: "<< T(a*2*nv+b,i*2*no+j) << vso[(a+tno)*tn3+(b+tno)*tn2+i*tn+j] << vso[(a+tno)*tn3+(b+tno)*tn2+j*tn+i] <<  TestMp2 << endl;
                    }
            cout << "Test MP2RR: "<< TestMp2 << endl;
        }
        */
    }
    
    
    bool delta(int p,int q)
    {
        if (p==q)
            return true;
        return false;
    }
    
    double CorrelatedEnergy(cx_mat& R2)
    {
        cx* vso = Vso.memptr();
        cx* r2 = R2.memptr();
        cx_mat Hso(Vso); Hso.zeros();
        cx_mat hso(Vso); hso.zeros();
        mat Xt = X.t();
        cx_mat Vt = V.t();
        mat h = (Xt)*(H)*X;
        cx_mat Htmp = Vt*h*V;
        hso.submat(0,0,n-1,n-1) = Htmp;
        hso.submat(n,n,tn-1,tn-1) = Htmp;
        
        cx_mat hmu = Vt*h*V;
        for (int p=0;p<tn;++p)
            for (int q=0;q<tn;++q)
                for (int r=0;r<tn;++r)
                    for (int s=0;s<tn;++s)
                    {
                        Hso[tn3*p+tn2*q+tn*r+s] = (1.0/(2.0*no-1.0))*(hso(p,r)*delta(q,s)+hso(q,s)*delta(p,r)-hso(p,s)*delta(q,r)-hso(q,r)*delta(p,s)) + vso[tn3*p+tn2*q+tn*r+s] - vso[tn3*p+tn2*q+tn*r+s];
                    }
        
        cx En(0.0,0.0);
        for (int p=0;p<tn;++p)
            for (int q=0;q<tn;++q)
                for (int r=0;r<tn;++r)
                    for (int s=0;s<tn;++s)
                        En += 0.25*(Hso[tn3*p+tn2*q+tn*r+s]*r2[tn3*r+tn2*s+tn*p+q]);
        cout << "Energy of Reduced hamiltonian: " << En + nuclear_energy << endl;
        return (real(En) + nuclear_energy);
    }
    
    // Argument is singlet spin-adapted.
    void Separabler2(const cx_mat Rho)
    {
        r2.resize(4*n2,4*n2);r2.zeros();
        cx_mat rso(tn,tn); rso.zeros();
        // Frozen density
        //cx_mat rso0(tn,tn); rso0.eye();
        //rso0.submat(no,no,n-1,n-1) *= 0.0;
        //rso0.submat(n+no,n+no,tn-1,tn-1) *= 0.0;
        // Up to Here
        rso.submat(0,0,n-1,n-1) = Rho;
        rso.submat(n,n,tn-1,tn-1) = Rho;
#pragma omp parallel for
        for (int p=0;p<tn;++p)
        {
            for (int q=0;q<tn;++q)
            {
                for (int r=0;r<tn;++r)
                {
                    for (int s=0;s<tn;++s)
                    {
                        r2[p*tn3+q*tn2+r*tn+s] = rso(p,r)*rso(q,s)-rso(q,r)*rso(p,s);
                    }
                }
            }
        }
        
        cx twotrace1(0.0,0.0),twotrace2(0.0,0.0);
        for (int p=0;p<tn;++p)
        {
            for (int q=0;q<tn;++q)
            {
                if(abs( r2[p*tn3+q*tn2+p*tn+q]+r2[p*tn3+q*tn2+q*tn+p])> 0.0000000001)
                    cout << "AntiSymm2?" << p << q << r2[p*tn3+q*tn2+p*tn+q] << r2[p*tn3+q*tn2+q*tn+p] << endl;
                twotrace1 += r2[p*tn3+q*tn2+p*tn+q];
                twotrace2 += r2[p*tn3+q*tn2+q*tn+p];
            }
        }
        if (params["Print"])
            cout << "2-traces2:" << twotrace1 << twotrace2 << endl;
        
    }
    
    // Rho is singlet spin-adapted.
    // Everything but the spin-orbital cumulant of R2 is replaced with grassman Rho.
    void TracedownConsistency(const cx_mat Rho,cx_mat& R2)
    {
        cx_mat rso(tn,tn); rso.zeros();
        rso.submat(0,0,n-1,n-1) = Rho;
        rso.submat(n,n,tn-1,tn-1) = Rho;
        cx_mat hso(tn,tn); hso.zeros();
        hso.submat(0,0,n-1,n-1).eye();
        hso.submat(n,n,tn-1,tn-1).eye();
        hso -= rso; // Hole density matrix.
        cx tt(0.0,0.0);
        // Force antisymmetry and hermiticity of the TBDM.
        R2 = 0.5*(R2+R2.t());
        for (int p=0;p<tn;++p)
        {
            for (int q=0;q<tn;++q)
            {
                tt += R2[p*tn3+q*tn2+p*tn+q];
                for (int r=0;r<tn;++r)
                {
                    for (int s=0;s<tn;++s)
                    {
                        cx tmp = 0.5*(R2[p*tn3+q*tn2+r*tn+s] - R2[p*tn3+q*tn2+s*tn+r]);
                        R2[p*tn3+q*tn2+r*tn+s] = tmp;
                        R2[p*tn3+q*tn2+s*tn+r] = -tmp;
                    }
                }
            }
        }
        // Correct R2.
        cx trrho = trace(rso);
        R2 *= (trrho-1)*(trrho)/(tt);
        // Force the correct trace.
        cout << "Tracedown" << trrho << tt << endl;
        cx_mat SP(rso); SP.zeros();
        // Tracedown R2.
        for (int p=0;p<tn;++p)
        {
            for (int q=0;q<tn;++q)
            {
                for (int r=0;r<tn;++r)
                {
                    SP[p*tn+q] += R2[p*tn3+r*tn2+q*tn+r];
                }
            }
        }
        // get effective n from SP.
        cx neff = 0.5*(1.0+sqrt(1.0+4.0*trace(SP)));
        cout << "nEff" << neff << endl;
        SP /= (neff-1.0);
        cout << "Tracedown2" << trace(SP) << endl;
        cx cumtrace(0.0,0.0); // Cumulant should trace to gamma*hole
        // Finally replace the separable part with grass.
        for (int p=0;p<tn;++p)
        {
            for (int q=0;q<tn;++q)
            {
                cumtrace += R2[p*tn3+q*tn2+p*tn+q] - (SP(p,p)*SP(q,q)-SP(q,p)*SP(p,q));
                /*
                 for (int r=0;r<tn;++r)
                 {
                 for (int s=0;s<tn;++s)
                 {
                 R2[p*tn3+q*tn2+r*tn+s] += rso(p,r)*rso(q,s)-rso(q,r)*rso(p,s) - (SP(p,r)*SP(q,s)-SP(q,r)*SP(p,s));
                 }
                 }
                 */
            }
        }
        cout << "Cumulant Trace? " << cumtrace << ":" << trace(-1.0*rso*hso) << endl;
        /*
         // Force antisymmetry, hermiticity of  TBDM.
         R2 = 0.5*(R2+R2.t());
         for (int p=0;p<tn;++p)
         {
         for (int q=0;q<tn;++q)
         {
         for (int r=0;r<tn;++r)
         {
         for (int s=0;s<tn;++s)
         {
         cx tmp = 0.5*(R2[p*tn3+q*tn2+r*tn+s] - R2[p*tn3+q*tn2+s*tn+r]);
         R2[p*tn3+q*tn2+r*tn+s] = tmp;
         R2[p*tn3+q*tn2+s*tn+r] = -tmp;
         }
         }
         }
         }*/
        
    }
    
    //    performs U.R2.Ut
    void Transform2(cx_mat& R2, const cx_mat& U)
    {
        cx_mat tmp(R2); tmp.zeros();
        cx_mat tmp2(R2); tmp2.zeros();
        // Contract over each dimension sequentially to get N(5)
#pragma omp parallel for
        for (int p=0;p<tn;++p)
        {
            for (int q=0;q<tn;++q)
            {
                for (int r=0;r<tn;++r)
                {
                    for (int s=0;s<tn;++s)
                    {
                        for (int t=0;t<tn;++t)
                        {
                            if (((p>=n) && (t<n)) || ((p<n) && (t>=n)))
                                continue;
                            tmp[p*tn3+q*tn2+r*tn+s] += U(p%n,t%n)*R2[t*tn3+q*tn2+r*tn+s];
                        }
                    }
                }
            }
        }
#pragma omp parallel for
        for (int p=0;p<tn;++p)
        {
            for (int q=0;q<tn;++q)
            {
                for (int r=0;r<tn;++r)
                {
                    for (int s=0;s<tn;++s)
                    {
                        for (int t=0;t<tn;++t)
                        {
                            if (((q>=n) && (t<n)) || ((q<n) && (t>=n)))
                                continue;
                            tmp2[p*tn3+q*tn2+r*tn+s] += U(q%n,t%n)*tmp[p*tn3+t*tn2+r*tn+s];
                        }
                    }
                }
            }
        }
        tmp.zeros();
#pragma omp parallel for
        for (int p=0;p<tn;++p)
        {
            for (int q=0;q<tn;++q)
            {
                for (int r=0;r<tn;++r)
                {
                    for (int s=0;s<tn;++s)
                    {
                        for (int t=0;t<tn;++t)
                        {
                            if (((r>=n) && (t<n)) || ((r<n) && (t>=n)))
                                continue;
                            tmp[p*tn3+q*tn2+r*tn+s] += tmp2[p*tn3+q*tn2+t*tn+s]*conj(U(r%n,t%n));
                        }
                    }
                }
            }
        }
        tmp2.zeros();
#pragma omp parallel for
        for (int p=0;p<tn;++p)
        {
            for (int q=0;q<tn;++q)
            {
                for (int r=0;r<tn;++r)
                {
                    for (int s=0;s<tn;++s)
                    {
                        for (int t=0;t<tn;++t)
                        {
                            if (((s>=n) && (t<n)) || ((s<n) && (t>=n)))
                                continue;
                            tmp2[p*tn3+q*tn2+r*tn+s] += tmp[p*tn3+q*tn2+r*tn+t]*conj(U(s%n,t%n));
                        }
                    }
                }
            }
        }
        R2 = tmp2;
        // Check some traces.
        /*        cx twotrace1(0.0,0.0),twotrace2(0.0,0.0);
         for (int p=0;p<tn;++p)
         {
         for (int q=0;q<tn;++q)
         {
         if(abs( R2[p*tn3+q*tn2+p*tn+q]+R2[p*tn3+q*tn2+q*tn+p])> 0.0000000001)
         cout << "AntiSymm?" << p << q << R2[p*tn3+q*tn2+p*tn+q] << R2[p*tn3+q*tn2+q*tn+p] << endl;
         twotrace1 += R2[p*tn3+q*tn2+p*tn+q];
         twotrace2 += R2[p*tn3+q*tn2+q*tn+p];
         }
         }
         cout << "2-Hrms:" << R2[1*tn3+2*tn2+3*tn+4] << R2[3*tn3+4*tn2+1*tn+2] << R2[3*tn3+4*tn2+2*tn+1] << R2[4*tn3+3*tn2+1*tn+2] << endl;
         cout << "2-traces:" << twotrace1 << twotrace2 << endl;
         */
    }
    
    int IsOcc(int a)
    {
        if( ((a > -1) && (a < no)) ||  ((a > n-1) && (a < n+no)))
            return 0;
        else if( ((a > no-1) && (a < n)) ||  ((a > n+no-1) && (a < n+n)))
            return 1;
        else
            return -1;
    }
    
    
    void sotdtda(cx_mat& Rho_, cx_mat& RhoDot_, const double time, cx_mat& hmu,bool& IsOn,FieldMatrices* Mus)
    {
        
        //cout << "checking sotdtda" << endl;
        cx_mat h = hmu;
        if(Mus!=NULL)
            Mus->ApplyField(h,time,IsOn);
        
        //Separabler2(Rho_);
        
        RhoDot_.zeros();
        cx_mat rso(tn,tn); rso.zeros();
        cx* vso = Vso.memptr();
        
        
        // Instead of Separabler2 TDTDA method
        cx_mat soRho(tn,tn); soRho.zeros();
        soRho.submat(0,0,n-1,n-1) = 0.5*Rho_;
        soRho.submat(n,n,tn-1,tn-1) = 0.5*Rho_;
        if(params["Print"])
            soRho.print("Rho");
        cx_mat soRho0(tn,tn); soRho0.eye();
        soRho0.submat(no,no,n-1,n-1) *= 0.0;
        soRho0.submat(n+no,n+no,tn-1,tn-1) *= 0.0;
        soRho0 *= 0.5;
        if(params["Print"])
            soRho0.print("Rho0");
        cx_mat tmp(rso); tmp.zeros();
        cx_mat ttmp(rso); ttmp.zeros();
        // Commutator
        // Not so Sure if this has to be in SO
        // Don't know how to make h in to SO, so it is in MO
        //cx_mat hso(soRho); hso.zeros();
        //hso.submat(0,0,n-1,n-1) = h;
        //hso.submat(n,n,tn-1,tn-1) = h;
        //tmp =(hso*soRho - soRho*hso);
        //RhoDot_ += -j*(tmp.submat(0,0,n-1,n-1)+tmp.submat(n,n,2*n-1,2*n-1));
        RhoDot_+= -j * (h*Rho_ - Rho_*h);
        
        
        //vso.print("VSO");
        
        // RPA motion
        tmp.zeros();
        rso.zeros();
#pragma omp parallel for
        for (int ap=0;ap<tn;++ap)
            for (int ip=0;ip<tn;++ip)
                for (int k=0;k<tn;++k)
                    for (int i=0;i<tn;++i)
                    {
                        // Alpha J and K
                        rso(ap,ip) += 0.5*(vso[tn3*ap+tn2*i+tn*ip+k]*soRho[tn*k+i]);
                        rso(ap,ip) -= 0.5*(vso[tn3*ap+tn2*i+tn*k+ip]*soRho[tn*k+i]);
                    }
        tmp = rso*soRho0 - soRho0*rso;
        if(params["Print"])
            tmp.print("WHOLE");
        RhoDot_ += -j*(tmp.submat(0,0,n-1,n-1)+tmp.submat(n,n,2*n-1,2*n-1));
        
        
        // Subtraction from RPA to form TDTDA
        // 0 = O, 1 = V
        int inup,indown,outup,outdown;
        
        // OV to VO subtraction
        inup=0; indown=1; outup=1; outdown=0;
        tmp.zeros();
        ttmp.zeros();
        rso.zeros();
#pragma omp parallel for
        for (int ap=0;ap<tn;++ap)
            for (int ip=0;ip<tn;++ip)
                for (int k=0;k<tn;++k)
                    for (int i=0;i<tn;++i)
                    {
                        if (!(IsOcc(k)==inup)) // is 0 if spin orbital k is occupied else 1
                            continue;
                        if (!(IsOcc(i)==indown))
                            continue;
                        rso(ap,ip) += 0.5*(vso[tn3*ap+tn2*i+tn*ip+k]*soRho[tn*k+i]);
                        rso(ap,ip) -= 0.5*(vso[tn3*ap+tn2*i+tn*k+ip]*soRho[tn*k+i]);
                    }
        ttmp = rso*soRho0 - soRho0*rso;
        if (outup==0 && outdown==0)
        {
            tmp.submat(0,0,no-1,no-1) = ttmp.submat(0,0,no-1,no-1);
            tmp.submat(n,n,n+no-1,n+no-1) = ttmp.submat(n,n,n+no-1,n+no-1);
        }
        if (outup==1 && outdown==0)
        {
            tmp.submat(no,0,n-1,no-1) = ttmp.submat(no,0,n-1,no-1);
            tmp.submat(n+no,n,tn-1,n+no-1) = ttmp.submat(n+no,n,tn-1,n+no-1);
        }
        if (outup==0 && outdown==1)
        {
            tmp.submat(0,no,no-1,n-1) = ttmp.submat(0,no,no-1,n-1);
            tmp.submat(n,n+no,n+no-1,tn-1) = ttmp.submat(n,n+no,n+no-1,tn-1);
        }
        if (outup==1 && outdown==1)
        {
            tmp.submat(no,no,n-1,n-1) = ttmp.submat(no,no,n-1,n-1);
            tmp.submat(n+no,n+no,tn-1,tn-1) = ttmp.submat(n+no,n+no,tn-1,tn-1);
        }
        if(params["Print"])
            tmp.print("OV TO VO");
        RhoDot_ -= -j*(tmp.submat(0,0,n-1,n-1)+tmp.submat(n,n,2*n-1,2*n-1));
        
        
        // VO to OV subtraction
        inup=1; indown=0; outup=0; outdown=1;
        tmp.zeros();
        ttmp.zeros();
        rso.zeros();
#pragma omp parallel for
        for (int ap=0;ap<tn;++ap)
            for (int ip=0;ip<tn;++ip)
                for (int k=0;k<tn;++k)
                    for (int i=0;i<tn;++i)
                    {
                        if (!(IsOcc(k)==inup)) // is 0 if spin orbital k is occupied else 1
                            continue;
                        if (!(IsOcc(i)==indown))
                            continue;
                        rso(ap,ip) += 0.5*(vso[tn3*ap+tn2*i+tn*ip+k]*soRho[tn*k+i]);
                        rso(ap,ip) -= 0.5*(vso[tn3*ap+tn2*i+tn*k+ip]*soRho[tn*k+i]);
                    }
        ttmp = rso*soRho0 - soRho0*rso;
        if (outup==0 && outdown==0)
        {
            tmp.submat(0,0,no-1,no-1) = ttmp.submat(0,0,no-1,no-1);
            tmp.submat(n,n,n+no-1,n+no-1) = ttmp.submat(n,n,n+no-1,n+no-1);
        }
        if (outup==1 && outdown==0)
        {
            tmp.submat(no,0,n-1,no-1) = ttmp.submat(no,0,n-1,no-1);
            tmp.submat(n+no,n,tn-1,n+no-1) = ttmp.submat(n+no,n,n+no-1,tn-1);
        }
        if (outup==0 && outdown==1)
        {
            tmp.submat(0,no,no-1,n-1) = ttmp.submat(0,no,no-1,n-1);
            tmp.submat(n,n+no,n+no-1,tn-1) = ttmp.submat(n,n+no,n+no-1,tn-1);
        }
        if (outup==1 && outdown==1)
        {
            tmp.submat(no,no,n-1,n-1) = ttmp.submat(no,no,n-1,n-1);
            tmp.submat(n+no,n+no,tn-1,tn-1) = ttmp.submat(n+no,n+no,tn-1,tn-1);
        }
        if(params["Print"])
            tmp.print("VO TO OV");
        RhoDot_ -= -j*(tmp.submat(0,0,n-1,n-1)+tmp.submat(n,n,2*n-1,2*n-1));
        
        
    }
    
    
    void tdtda(cx_mat& Rho_, cx_mat& RhoDot_, const double time, cx_mat& hmu,bool& IsOn,FieldMatrices* Mus)
    {
        
        cx_mat h = hmu;
        // Apply the field
        if(Mus!=NULL)
            Mus->ApplyField(h,time,IsOn);
        
        
#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
#endif
        const cx* Vi = IntCur.memptr();
        cx_mat tmp(RhoDot_); tmp.zeros();
        cx_mat ttmp(RhoDot_); ttmp.zeros();
        cx_mat JK(RhoDot_); JK.zeros();
        cx_mat J(RhoDot_); J.zeros();
        cx_mat K(RhoDot_); K.zeros();
        int inup, indown, outup, outdown;
        
        // Set up Rho(t) and Rho(0)
        cx_mat Rho0(Rho_); Rho0.eye();
        Rho0.submat(no,no,n-1,n-1).zeros();
        cx_mat Rho1(Rho_);
        
        
        RhoDot_+= -j * (h*Rho1 - Rho1*h);
        
        if (params["Print"])
            RhoDot_.print("\nCore Hamiltonian part");
        
#if HASUDR
#pragma omp parallel for reduction(+: J,K) schedule(static)
#endif
        for (int ip=0; ip<n; ++ip)
            for (int i=0; i<n; ++i)
            {
                for (int ap=0; ap<n; ++ap)
                    for (int k=0; k<n; ++k)
                    {
                        J[ap*n+ip] += -j*2.0*(Vi[ap*n3+i*n2+ip*n+k])*Rho0[k*n+i];
                        K[ap*n+ip] -= -j*(Vi[ap*n3+i*n2+k*n+ip])*Rho0[k*n+i];
                    }
            }
        tmp=(J+K)*Rho1 - Rho1*(J+K); // h + this is the fock part.
        if (params["Print"])
            tmp.print("+ alpha (to core hamiltonian)");
        RhoDot_ +=  tmp;
        
        
        // VO TO VO AND OV TO OV
        inup=1; indown=0; outup=1; outdown=0;
        tmp.zeros();
        ttmp.zeros();
        J.zeros(); K.zeros();
#if HASUDR
#pragma omp parallel for reduction(+: J,K) schedule(static)
#endif
        for (int ap=0;ap<n;++ap)
            for (int ip=0;ip<n;++ip)
                for (int k=0;k<n;++k)
                    for (int i=0;i<n;++i)
                    {
                        if (!(IsOcc(i)==inup)) // is 0 if spin orbital k is occupied else 1
                            continue;
                        if (!(IsOcc(k)==indown))
                            continue;
                        J[ap*n+ip] += -j*2.0*(Vi[ap*n3+i*n2+ip*n+k])*Rho1[k*n+i];
                        K[ap*n+ip] -= -j*(Vi[ap*n3+i*n2+k*n+ip])*Rho1[k*n+i];
                    }
        ttmp = (J+K)*Rho0 - Rho0*(J+K);
        if (outup==0 && outdown==0)
            tmp.submat(0,0,no-1,no-1) = ttmp.submat(0,0,no-1,no-1);
        if (outup==1 && outdown==0)
            tmp.submat(no,0,n-1,no-1) = ttmp.submat(no,0,n-1,no-1);
        if (outup==0 && outdown==1)
            tmp.submat(0,no,no-1,n-1) = ttmp.submat(0,no,no-1,n-1);
        if (outup==1 && outdown==1)
            tmp.submat(no,no,n-1,n-1) = ttmp.submat(no,no,n-1,n-1);
        if(params["Print"])
            tmp.print("VO TO VO");
        RhoDot_ += tmp;
       
        
        inup=0; indown=1; outup=0; outdown=1;
        tmp.zeros();
        ttmp.zeros();
        J.zeros(); K.zeros();
#if HASUDR
#pragma omp parallel for reduction(+: J,K) schedule(static)
#endif
        for (int ap=0;ap<n;++ap)
            for (int ip=0;ip<n;++ip)
                for (int k=0;k<n;++k)
                    for (int i=0;i<n;++i)
                    {
                        if (!(IsOcc(i)==inup)) // is 0 if spin orbital k is occupied else 1
                            continue;
                        if (!(IsOcc(k)==indown))
                            continue;
                        J[ap*n+ip] += -j*2.0*(Vi[ap*n3+i*n2+ip*n+k])*Rho1[k*n+i];
                        K[ap*n+ip] -= -j*(Vi[ap*n3+i*n2+k*n+ip])*Rho1[k*n+i];
                    }
        ttmp = (J+K)*Rho0 - Rho0*(J+K);
        if (outup==0 && outdown==0)
            tmp.submat(0,0,no-1,no-1) = ttmp.submat(0,0,no-1,no-1);
        if (outup==1 && outdown==0)
            tmp.submat(no,0,n-1,no-1) = ttmp.submat(no,0,n-1,no-1);
        if (outup==0 && outdown==1)
            tmp.submat(0,no,no-1,n-1) = ttmp.submat(0,no,no-1,n-1);
        if (outup==1 && outdown==1)
            tmp.submat(no,no,n-1,n-1) = ttmp.submat(no,no,n-1,n-1);
        if(params["Print"])
            tmp.print("OV TO OV");
        RhoDot_ += tmp;
        
        /*
        // OO to OFF
        inup=0; indown=0; outup=0; outdown=1;
        tmp.zeros();
        ttmp.zeros();
        J.zeros(); K.zeros();
#if HASUDR
#pragma omp parallel for reduction(+: J,K) schedule(static)
#endif
        for (int ap=0;ap<n;++ap)
            for (int ip=0;ip<n;++ip)
                for (int k=0;k<n;++k)
                    for (int i=0;i<n;++i)
                    {
                        if (!(IsOcc(i)==inup)) // is 0 if spin orbital k is occupied else 1
                            continue;
                        if (!(IsOcc(k)==indown))
                            continue;
                        J[ap*n+ip] += -j*2.0*(Vi[ap*n3+i*n2+ip*n+k])*Rho1[k*n+i];
                        K[ap*n+ip] -= -j*(Vi[ap*n3+i*n2+k*n+ip])*Rho1[k*n+i];
                    }
        ttmp = (J+K)*Rho1 - Rho1*(J+K);
        if (outup==0 && outdown==0)
            tmp.submat(0,0,no-1,no-1) = ttmp.submat(0,0,no-1,no-1);
        if (outup==1 && outdown==0)
            tmp.submat(no,0,n-1,no-1) = ttmp.submat(no,0,n-1,no-1);
        if (outup==0 && outdown==1)
            tmp.submat(0,no,no-1,n-1) = ttmp.submat(0,no,no-1,n-1);
        if (outup==1 && outdown==1)
            tmp.submat(no,no,n-1,n-1) = ttmp.submat(no,no,n-1,n-1);
        if(params["Print"])
            tmp.print("VO TO VO");
        RhoDot_ += tmp;
        
        
        inup=0; indown=0; outup=1; outdown=0;
        tmp.zeros();
        ttmp.zeros();
        J.zeros(); K.zeros();
#if HASUDR
#pragma omp parallel for reduction(+: J,K) schedule(static)
#endif
        for (int ap=0;ap<n;++ap)
            for (int ip=0;ip<n;++ip)
                for (int k=0;k<n;++k)
                    for (int i=0;i<n;++i)
                    {
                        if (!(IsOcc(i)==inup)) // is 0 if spin orbital k is occupied else 1
                            continue;
                        if (!(IsOcc(k)==indown))
                            continue;
                        J[ap*n+ip] += -j*2.0*(Vi[ap*n3+i*n2+ip*n+k])*Rho1[k*n+i];
                        K[ap*n+ip] -= -j*(Vi[ap*n3+i*n2+k*n+ip])*Rho1[k*n+i];
                    }
        ttmp = (J+K)*Rho1 - Rho1*(J+K);
        if (outup==0 && outdown==0)
            tmp.submat(0,0,no-1,no-1) = ttmp.submat(0,0,no-1,no-1);
        if (outup==1 && outdown==0)
            tmp.submat(no,0,n-1,no-1) = ttmp.submat(no,0,n-1,no-1);
        if (outup==0 && outdown==1)
            tmp.submat(0,no,no-1,n-1) = ttmp.submat(0,no,no-1,n-1);
        if (outup==1 && outdown==1)
            tmp.submat(no,no,n-1,n-1) = ttmp.submat(no,no,n-1,n-1);
        if(params["Print"])
            tmp.print("OV TO OV");
        RhoDot_ += tmp;
        
        // VV to OFF
        inup=1; indown=1; outup=0; outdown=1;
        tmp.zeros();
        ttmp.zeros();
        J.zeros(); K.zeros();
#if HASUDR
#pragma omp parallel for reduction(+: J,K) schedule(static)
#endif
        for (int ap=0;ap<n;++ap)
            for (int ip=0;ip<n;++ip)
                for (int k=0;k<n;++k)
                    for (int i=0;i<n;++i)
                    {
                        if (!(IsOcc(i)==inup)) // is 0 if spin orbital k is occupied else 1
                            continue;
                        if (!(IsOcc(k)==indown))
                            continue;
                        J[ap*n+ip] += -j*2.0*(Vi[ap*n3+i*n2+ip*n+k])*Rho1[k*n+i];
                        K[ap*n+ip] -= -j*(Vi[ap*n3+i*n2+k*n+ip])*Rho1[k*n+i];
                    }
        ttmp = (J+K)*Rho1 - Rho1*(J+K);
        if (outup==0 && outdown==0)
            tmp.submat(0,0,no-1,no-1) = ttmp.submat(0,0,no-1,no-1);
        if (outup==1 && outdown==0)
            tmp.submat(no,0,n-1,no-1) = ttmp.submat(no,0,n-1,no-1);
        if (outup==0 && outdown==1)
            tmp.submat(0,no,no-1,n-1) = ttmp.submat(0,no,no-1,n-1);
        if (outup==1 && outdown==1)
            tmp.submat(no,no,n-1,n-1) = ttmp.submat(no,no,n-1,n-1);
        if(params["Print"])
            tmp.print("VO TO VO");
        RhoDot_ += tmp;
        
        
        inup=1; indown=1; outup=1; outdown=0;
        tmp.zeros();
        ttmp.zeros();
        J.zeros(); K.zeros();
#if HASUDR
#pragma omp parallel for reduction(+: J,K) schedule(static)
#endif
        for (int ap=0;ap<n;++ap)
            for (int ip=0;ip<n;++ip)
                for (int k=0;k<n;++k)
                    for (int i=0;i<n;++i)
                    {
                        if (!(IsOcc(i)==inup)) // is 0 if spin orbital k is occupied else 1
                            continue;
                        if (!(IsOcc(k)==indown))
                            continue;
                        J[ap*n+ip] += -j*2.0*(Vi[ap*n3+i*n2+ip*n+k])*Rho1[k*n+i];
                        K[ap*n+ip] -= -j*(Vi[ap*n3+i*n2+k*n+ip])*Rho1[k*n+i];
                    }
        ttmp = (J+K)*Rho1 - Rho1*(J+K);
        if (outup==0 && outdown==0)
            tmp.submat(0,0,no-1,no-1) = ttmp.submat(0,0,no-1,no-1);
        if (outup==1 && outdown==0)
            tmp.submat(no,0,n-1,no-1) = ttmp.submat(no,0,n-1,no-1);
        if (outup==0 && outdown==1)
            tmp.submat(0,no,no-1,n-1) = ttmp.submat(0,no,no-1,n-1);
        if (outup==1 && outdown==1)
            tmp.submat(no,no,n-1,n-1) = ttmp.submat(no,no,n-1,n-1);
        if(params["Print"])
            tmp.print("OV TO OV");
        RhoDot_ += tmp;
        */
        return;
        
    }
    
    
    // Depends on current r2.
    // Argument is singlet spin-adapted.
    // It is converted to spin-orbital here, and then finally shifted back.
    void bbgky1(cx_mat& Rho, cx_mat& RhoDot, const double time)
    {
        
//        if (params["TDTDA"])
//        {
//            tdtda(Rho,RhoDot);
//            return;
//        }
        
        if (params["BBGKY1"])
            Separabler2(Rho);//r2 built
        
        RhoDot.zeros();
        cx_mat rso(tn,tn); rso.zeros();
        cx* vso = Vso.memptr();
        
        rso.zeros();
#pragma omp parallel for
        for (int p=0;p<tn;++p)
        {
            for (int r=0;r<tn;++r)
                for (int s=0;s<tn;++s)
                    for (int t=0;t<tn;++t)
                    {
                        for (int q=0;q<tn;++q)
                        {
                            rso(p,q) += 0.5*(vso[tn3*p+tn2*r+tn*s+t]*r2[tn3*s+tn2*t+tn*q+r]);
                            rso(p,q) -= 0.5*(r2[tn3*p+tn2*r+tn*s+t]*vso[tn3*s+tn2*t+tn*q+r]);
                        }
                    }
        }
        
        RhoDot = -j*(rso.submat(0,0,n-1,n-1)+rso.submat(n,n,2*n-1,2*n-1));
    }
    
    // Depends on current r2 and vso (spin-orbital V -- unsymmetrized)
    // r2 also has to be transformed at each step.
    // These matrices are held in the spin-orbital basis
    // r2 should be transformed ahead of this routine.
    void bbgky2(cx_mat& R2, cx_mat& r2d, const double t)
    {
        r2d.zeros();
        cx* vso = Vso.memptr();
        /*
         // Done exponentially.
         for (int p=0;p<tn;++p)
         for (int q=0;q<tn;++q)
         for (int r=0;r<tn;++r)
         for (int s=0;s<tn;++s)
         {
         r2d[tn3*p+tn2*q+tn*r+s] = -j*(R2[tn3*p+tn2*q+tn*r+s]*(eigs(p%n)+eigs(q%n)-eigs(s%n)-eigs(r%n)));
         }
         */
        // two body terms.
#pragma omp parallel for
        for (int t=0;t<tn;++t)
        {
            for (int p=0;p<tn;++p)
            {
                for (int q=0;q<tn;++q)
                    for (int r=0;r<tn;++r)
                        for (int s=0;s<tn;++s)
                        {
                            if (R2[tn3*p+tn2*q+tn*r+s]==0.0)
                                continue;
                            for (int u=0;u<tn;++u)
                            {
                                r2d[tn3*t+tn2*u+tn*r+s] += -j*(vso[tn3*t+tn2*u+tn*p+q]*R2[tn3*p+tn2*q+tn*r+s]);
                                r2d[tn3*p+tn2*q+tn*t+u] -= -j*(R2[tn3*p+tn2*q+tn*r+s]*vso[tn3*r+tn2*s+tn*t+u]);
                            }
                        }
            }
        }
        r2d = 0.5*(r2d+r2d.t());
    }
    
    inline cx g0(const cx_mat& d2,const cx_mat& ed2, const int& p, const int& q, const int& r, const int& s, int type=0)
    {
        cx d = (d2(p,r)+d2(q,s));
        if (abs(d)<0.000000001 or type==-1)
            return cx(0.0,0.0);
        else if (type==1)
            (j/(j*0.00000001+d));
        else if (type==2) // Full term.
            return j*((ed2(p,r)*ed2(q,s))-1.0)/d;
        else if (type==3) // The inhomogeniety
            return -j*((ed2(p,r)*ed2(q,s)))/d;
        else if (type==0)
            return -j/d;
    }
    
    cx_mat FockBuildNoRotation(const cx_mat& Rho_, cx_mat& hmu, const double& time, double epsilon=0.000000001)
    {
        cx_mat Rho(Rho_);
        cx_mat Vfock(n,n); Vfock.zeros();
        cx* Vi = IntCur.memptr();
        for (int p=0;p<n;++p)
        {
            for (int q=0;q<n;++q)
            {
                for(int r=0; r<n; ++r)
                {
                    for(int s=0; s<n; ++s)
                    {
                        Vfock(p,q) += 2.0*Vi[r*n3+p*n2+s*n+q]*Rho(s,r);
                        Vfock(p,q) -= Vi[p*n3+r*n2+s*n+q]*Rho(s,r);
                    }
                }
            }
        }
        //RhoDot_ -= -j*(Vfock*Rho_-Rho_*Vfock);
        hmu += Vfock;
        threading_policy::pop();
        return hmu;
    }
    
    double MeanFieldEnergy(const cx_mat& Rho_)
    {
        
        mat Xt = X.t();
        cx_mat Vt = V.t();
        cx_mat h = Vt*(Xt)*(H)*X*V;
        Ehf = real(dot(Rho_,h));
        double t=0;
        FockBuildNoRotation(Rho_, h, t);
        Ehf += real(dot(Rho_,h));
        return Ehf+nuclear_energy;
    }
    
    cx RhoDot(const cx_mat& Rho_, cx_mat& RhoDot_, const double& time)
    {
        if (params["RIFock"])
            return (trace(RhoDot_%Rho_));
        else if (params["TDCIS"])
            RhoDotCIS(Rho_, RhoDot_);
        else if (params["2Born"])
            RhoDot2B(Rho_, RhoDot_, time);
        else if (params["Corr"])
            RhoDotCorr(Rho_, RhoDot_, time);
        if (tcl!=NULL)
            tcl->ContractGammaRhoMThreads(RhoDot_,Rho_);
        return (trace(RhoDot_%Rho_));
        
    }
    
    void RhoDotCIS(const cx_mat& Rho_, cx_mat& RhoDot_) const
    {
        
#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
#endif
        const cx* Vi = IntCur.memptr();
        cx_mat tmp(RhoDot_); tmp.zeros();
#if HASUDR
#pragma omp parallel for reduction(+: tmp) schedule(static)
#endif
        for (int i=0; i<no; ++i)
        {
            for (int a=no; a<n; ++a)
                if (abs(Rho_[a*n+i]) > 0.0000000001)
                    for (int ap=no; ap<n; ++ap)
                        for (int ip=0; ip<no; ++ip)
                        {
                            tmp[ap*n+ip] += j*(2.0*Vi[ap*n3+i*n2+ip*n+a]-Vi[ap*n3+i*n2+a*n+ip])*Rho_[a*n+i];
                        }
        }
        RhoDot_ += tmp+tmp.t();
        return;
    }
    
    cx RhoDotCorr(const cx_mat& Rho_, cx_mat& RhoDot_, const double& time, double epsilon=0.000000001, int g0flag=2)
    {
        cx_mat Rho(Rho_);
        if (g0flag!=3 && !params["NoInhom"]) // Call the inhomogeneity.
        {
            RhoDotCorr(Rho_, RhoDot_, time, epsilon, 3);
        }
        else if (g0flag==3 && !params["TDCIS"])// This is the inhomogeneity.
        {
            Rho = (V.t())*rho0*V;
        }
        
        for (int i=0; i<n; ++i)
        {
            if (real(Rho(i,i))<0)
                Rho(i,i)=0.0;
            if( abs(Rho(i,i))>1.0)
                Rho(i,i)=1.0;
        }
        
#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
        
#pragma omp declare reduction( + : cx_vec : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
#endif
        
        if (params["RIFock"]!=0.0)
            return 0.0;
        
        threading_policy::enable_omp_only();
        
        cx_mat Eta(Rho); Eta.eye();
        Eta -= Rho;
        
        cx_vec rho(Rho.diag());
        cx_vec eta(Eta.diag());
        
        cx_mat d2 = Delta2();
        cx_mat ed2 = exp(-j*d2*time);
        const cx* Vi = IntCur.memptr();
        cx_mat VI_ss(n2,n2); cx* Vi_ss = VI_ss.memptr();
        
        // Mp2 part.
        if (g0flag!=3 && params["IncMP2"])
        {
            /*
             cx_mat t(n2,n2); t.zeros();
             cx* T = t.memptr();
             Emp2=0.0;
             for (int i=0; i<n; ++i)
             {
             for (int j=0; j<n; ++j)
             {
             for (int a=0; a<n; ++a)
             {
             for (int b=0; b<n; ++b)
             {
             cx d = d2(a,i)+d2(b,j); //(eigs(a)*eta(a)+eigs(b)*eta(b)-eigs(i)*rho(i)-eigs(j)*rho(j));
             if (abs(d) > 0.0000001)
             {
             T[a*n3+b*n2+i*n+j] = (2*Vi[i*n3+j*n2+a*n+b] - Vi[i*n3+j*n2+b*n+a])/d;
             Emp2 += real(T[a*n3+b*n2+i*n+j]*Vi[i*n3+j*n2+a*n+b]*rho(i)*rho(j)*eta(a)*eta(b));
             }
             }
             }
             }
             }
             cout << "Emp2: " << Emp2 << endl;
             */
            
            cx_mat mp2(Rho.n_rows,Rho.n_cols); mp2.zeros();
            mp2 += j*Emp2;
            cx_mat Mp2 = symmatu(mp2,true);
            Mp2.diag().zeros();
            //                Mp2.print("mp2");
            RhoDot_ += Mp2%Rho;
        }
        
        if(1)
        {
            cx_mat G0(n2,n2); cx* g0 = G0.memptr();
            
#pragma omp parallel for
            for(int p=0; p<n; ++p)
            {
                for(int q=0; q<n; ++q)
                {
                    for(int r=0; r<n; ++r)
                    {
                        for(int s=0; s<n; ++s)
                        {
                            Vi_ss[p*n3+q*n2+r*n+s] = Vi[p*n3+q*n2+r*n+s] - Vi[p*n3+q*n2+s*n+r];
                            cx d = (d2(p,r)+d2(q,s));//(eigs(p)*rho(p)+eigs(q)*rho(q)-eigs(r)*eta(r)-eigs(s)*eta(s));
                            if (abs(d) > 0.0000000001)
                            {
                                if (g0flag == 2)
                                    g0[p*n3+q*n2+r*n+s] = j*((ed2(p,r)*ed2(q,s))-1.0)/d;
                                else
                                    g0[p*n3+q*n2+r*n+s] = -j*((ed2(p,r)*ed2(q,s)))/d;
                            }
                            else
                                g0[p*n3+q*n2+r*n+s] = 0.0;
                        }
                    }
                }
            }
            
            
            cx_mat Out(RhoDot_); Out.zeros();
            cx_mat Oute(RhoDot_); Oute.zeros();
            cx_mat Outr(RhoDot_); Outr.zeros();
            cx_mat Oute2(RhoDot_); Oute2.zeros();
            cx_mat Outr2(RhoDot_); Outr2.zeros();
            
#pragma omp parallel for
            for(int p=0; p<n; ++p)
            {
                for(int r=0; r<n; ++r)
                {
                    for(int s=0; s<n; ++s)
                    {
                        for(int t=0; t<n; ++t)
                        {
                            Oute(p,p) += -1.*Vi_ss[p*n3+r*n2+s*n+t]*g0[p*n3+r*n2+s*n+t]*Vi_ss[s*n3+t*n2+p*n+r]*Rho(t,s)*Rho(s,t)*eta(r);
                            Oute(p,p) += 1.*Vi[p*n3+r*n2+s*n+t]*g0[p*n3+r*n2+s*n+t]*Vi[s*n3+t*n2+p*n+r]*rho(s)*rho(t)*eta(r);
                            Oute(p,p) += 1.*Vi[p*n3+r*n2+t*n+s]*g0[p*n3+r*n2+t*n+s]*Vi[t*n3+s*n2+p*n+r]*rho(s)*rho(t)*eta(r);
                            Oute(p,p) += 1.*Vi_ss[p*n3+r*n2+s*n+t]*g0[p*n3+r*n2+s*n+t]*Vi_ss[s*n3+t*n2+p*n+r]*rho(s)*rho(t)*eta(r);
                            Outr(p,t) += -1.*Vi_ss[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+t]*Rho(p,t)*Eta(s,r)*Eta(r,s);
                            Outr(p,p) += 1.*Vi_ss[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+t]*rho(t)*Eta(s,r)*Eta(r,s);
                            Outr(p,t) += 1.*Vi_ss[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+t]*Rho(p,t)*eta(r)*eta(s);
                            Outr(p,p) += -1.*Vi[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi[r*n3+s*n2+p*n+t]*rho(t)*eta(r)*eta(s);
                            Outr(p,p) += -1.*Vi[p*n3+t*n2+s*n+r]*g0[p*n3+t*n2+s*n+r]*Vi[s*n3+r*n2+p*n+t]*rho(t)*eta(r)*eta(s);
                            Outr(p,p) += -1.*Vi_ss[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+t]*rho(t)*eta(r)*eta(s);
                            Oute(p,t) += 1.*Vi_ss[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+t]*Rho(s,r)*Rho(r,s)*Eta(p,t);
                            Oute(p,t) += -1.*Vi_ss[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+t]*rho(r)*rho(s)*Eta(p,t);
                            
                            
                        }
                    }
                }
            }
            
#pragma omp parallel for
            for(int q=0; q<n; ++q)
            {
                for(int r=0; r<n; ++r)
                {
                    for(int s=0; s<n; ++s)
                    {
                        for(int t=0; t<n; ++t)
                        {
                            Oute2(q,q) += -1.*Vi_ss[s*n3+t*n2+q*n+r]*g0[s*n3+t*n2+q*n+r]*Vi_ss[q*n3+r*n2+s*n+t]*Rho(t,s)*Rho(s,t)*eta(r);
                            Oute2(q,q) += 1.*Vi[s*n3+t*n2+q*n+r]*g0[s*n3+t*n2+q*n+r]*Vi[q*n3+r*n2+s*n+t]*rho(s)*rho(t)*eta(r);
                            Oute2(q,q) += 1.*Vi[t*n3+s*n2+q*n+r]*g0[t*n3+s*n2+q*n+r]*Vi[q*n3+r*n2+t*n+s]*rho(s)*rho(t)*eta(r);
                            Oute2(q,q) += 1.*Vi_ss[s*n3+t*n2+q*n+r]*g0[s*n3+t*n2+q*n+r]*Vi_ss[q*n3+r*n2+s*n+t]*rho(s)*rho(t)*eta(r);
                            Outr2(t,q) += -1.*Vi_ss[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi_ss[q*n3+t*n2+r*n+s]*Rho(t,q)*Eta(s,r)*Eta(r,s);
                            Outr2(q,q) += 1.*Vi_ss[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi_ss[q*n3+t*n2+r*n+s]*rho(t)*Eta(s,r)*Eta(r,s);
                            Outr2(t,q) += 1.*Vi_ss[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi_ss[q*n3+t*n2+r*n+s]*Rho(t,q)*eta(r)*eta(s);
                            Outr2(q,q) += -1.*Vi[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi[q*n3+t*n2+r*n+s]*rho(t)*eta(r)*eta(s);
                            Outr2(q,q) += -1.*Vi[s*n3+r*n2+q*n+t]*g0[s*n3+r*n2+q*n+t]*Vi[q*n3+t*n2+s*n+r]*rho(t)*eta(r)*eta(s);
                            Outr2(q,q) += -1.*Vi_ss[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi_ss[q*n3+t*n2+r*n+s]*rho(t)*eta(r)*eta(s);
                            Oute2(t,q) += 1.*Vi_ss[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi_ss[q*n3+t*n2+r*n+s]*Rho(s,r)*Rho(r,s)*Eta(t,q);
                            Oute2(t,q) += -1.*Vi_ss[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi_ss[q*n3+t*n2+r*n+s]*rho(r)*rho(s)*Eta(t,q);
                            
                        }
                    }
                }
            }
            
            Out += Oute*Eta;
            Out += Outr*Rho;
            Out += Eta*Oute2;
            Out += Rho*Outr2;
            
            RhoDot_ += 0.5*(Out+Out.t());
            
        }
        
        if(0)
        {
            cx_mat G0(n2,n2); cx* g0 = G0.memptr();
            
#pragma omp parallel for
            for(int p=0; p<n; ++p)
            {
                for(int q=0; q<n; ++q)
                {
                    for(int r=0; r<n; ++r)
                    {
                        for(int s=0; s<n; ++s)
                        {
                            Vi_ss[p*n3+q*n2+r*n+s] = Vi[p*n3+q*n2+r*n+s] - Vi[p*n3+q*n2+s*n+r];
                            cx d = (d2(p,r)+d2(q,s));//(eigs(p)*rho(p)+eigs(q)*rho(q)-eigs(r)*eta(r)-eigs(s)*eta(s));
                            if (abs(d) > 0.0000000001)
                            {
                                if (g0flag == 2)
                                    g0[p*n3+q*n2+r*n+s] = j*((ed2(p,r)*ed2(q,s))-1.0)/d;
                                else
                                    g0[p*n3+q*n2+r*n+s] = -j*((ed2(p,r)*ed2(q,s)))/d;
                            }
                            else
                                g0[p*n3+q*n2+r*n+s] = 0.0;
                        }
                    }
                }
            }
            
            
            cx_mat Out(RhoDot_); Out.zeros();
            cx_mat Oute(RhoDot_); Oute.zeros();
            cx_mat Outr(RhoDot_); Outr.zeros();
            cx_mat Oute2(RhoDot_); Oute2.zeros();
            cx_mat Outr2(RhoDot_); Outr2.zeros();
            
#pragma omp parallel for
            for(int p=0; p<n; ++p)
            {
                for(int r=0; r<n; ++r)
                {
                    for(int s=0; s<n; ++s)
                    {
                        for(int t=0; t<n; ++t)
                        {
                            Oute(p,p) += -1.*Vi_ss[p*n3+r*n2+s*n+t]*g0[p*n3+r*n2+s*n+t]*Vi_ss[s*n3+t*n2+p*n+r]*Rho(t,s)*Rho(s,t)*eta(r);
                            Oute(p,p) += 1.*Vi[p*n3+r*n2+s*n+t]*g0[p*n3+r*n2+s*n+t]*Vi[s*n3+t*n2+p*n+r]*rho(s)*rho(t)*eta(r);
                            Oute(p,p) += 1.*Vi[p*n3+r*n2+t*n+s]*g0[p*n3+r*n2+t*n+s]*Vi[t*n3+s*n2+p*n+r]*rho(s)*rho(t)*eta(r);
                            Oute(p,p) += 1.*Vi_ss[p*n3+r*n2+s*n+t]*g0[p*n3+r*n2+s*n+t]*Vi_ss[s*n3+t*n2+p*n+r]*rho(s)*rho(t)*eta(r);
                            Oute(p,s) += 2.*Vi_ss[p*n3+r*n2+p*n+t]*g0[p*n3+r*n2+p*n+t]*Vi_ss[s*n3+t*n2+r*n+s]*Rho(t,s)*Rho(p,t)*eta(r);
                            Oute(p,t) += 2.*Vi[p*n3+r*n2+p*n+s]*g0[p*n3+r*n2+p*n+s]*Vi[t*n3+s*n2+t*n+r]*rho(s)*Rho(p,t)*eta(r);
                            Oute(p,t) += 2.*Vi_ss[p*n3+r*n2+p*n+s]*g0[p*n3+r*n2+p*n+s]*Vi_ss[s*n3+t*n2+r*n+t]*rho(s)*Rho(p,t)*eta(r);
                            Oute(p,p) += -2.*Vi_ss[p*n3+r*n2+r*n+t]*g0[p*n3+r*n2+r*n+t]*Vi_ss[s*n3+t*n2+p*n+s]*Rho(t,s)*Rho(r,t)*Eta(s,r);
                            Oute(p,p) += 2.*Vi[p*n3+r*n2+t*n+r]*g0[p*n3+r*n2+t*n+r]*Vi[t*n3+s*n2+p*n+s]*Rho(r,s)*rho(t)*Eta(s,r);
                            Oute(p,p) += 2.*Vi_ss[p*n3+r*n2+r*n+t]*g0[p*n3+r*n2+r*n+t]*Vi_ss[s*n3+t*n2+p*n+s]*Rho(r,s)*rho(t)*Eta(s,r);
                            Oute(p,r) += -2.*Vi_ss[p*n3+r*n2+p*n+t]*g0[p*n3+r*n2+p*n+t]*Vi_ss[s*n3+t*n2+r*n+s]*Rho(t,s)*Rho(p,t)*Eta(s,r);
                            Oute(p,r) += 2.*Vi_ss[p*n3+r*n2+p*n+t]*g0[p*n3+r*n2+p*n+t]*Vi_ss[s*n3+t*n2+r*n+s]*Rho(p,s)*rho(t)*Eta(s,r);
                            Oute(p,r) += -1.*Vi_ss[p*n3+s*n2+s*n+t]*g0[p*n3+s*n2+s*n+t]*Vi_ss[r*n3+t*n2+p*n+r]*Rho(s,r)*rho(t)*Eta(p,s);
                            Oute(p,t) += 2.*Vi_ss[p*n3+s*n2+r*n+s]*g0[p*n3+s*n2+r*n+s]*Vi_ss[r*n3+t*n2+p*n+t]*Rho(s,r)*Rho(r,t)*Eta(p,s);
                            Outr(p,t) += -2.*Vi_ss[p*n3+t*n2+p*n+s]*g0[p*n3+t*n2+p*n+s]*Vi_ss[r*n3+s*n2+r*n+t]*Rho(r,t)*Eta(s,r)*Eta(p,s);
                            Outr(p,r) += 2.*Vi_ss[p*n3+t*n2+p*n+s]*g0[p*n3+t*n2+p*n+s]*Vi_ss[r*n3+s*n2+r*n+t]*rho(t)*Eta(s,r)*Eta(p,s);
                            Outr(p,t) += -1.*Vi_ss[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+t]*Rho(p,t)*Eta(s,r)*Eta(r,s);
                            Outr(p,p) += 1.*Vi_ss[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+t]*rho(t)*Eta(s,r)*Eta(r,s);
                            Outr(p,t) += 1.*Vi_ss[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+t]*Rho(p,t)*eta(r)*eta(s);
                            Outr(p,p) += -1.*Vi[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi[r*n3+s*n2+p*n+t]*rho(t)*eta(r)*eta(s);
                            Outr(p,p) += -1.*Vi[p*n3+t*n2+s*n+r]*g0[p*n3+t*n2+s*n+r]*Vi[s*n3+r*n2+p*n+t]*rho(t)*eta(r)*eta(s);
                            Outr(p,p) += -1.*Vi_ss[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+t]*rho(t)*eta(r)*eta(s);
                            Oute(p,s) += -1.*Vi_ss[p*n3+t*n2+r*n+t]*g0[p*n3+t*n2+r*n+t]*Vi_ss[r*n3+s*n2+p*n+s]*rho(r)*Rho(t,s)*Eta(p,t);
                            Oute(p,t) += 1.*Vi_ss[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+t]*Rho(s,r)*Rho(r,s)*Eta(p,t);
                            Oute(p,t) += -1.*Vi_ss[p*n3+t*n2+r*n+s]*g0[p*n3+t*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+t]*rho(r)*rho(s)*Eta(p,t);
                            Outr(p,t) += -2.*Vi[p*n3+s*n2+p*n+r]*g0[p*n3+s*n2+p*n+r]*Vi[t*n3+r*n2+t*n+s]*rho(s)*eta(r)*Eta(p,t);
                            Outr(p,t) += -2.*Vi_ss[p*n3+s*n2+p*n+r]*g0[p*n3+s*n2+p*n+r]*Vi_ss[r*n3+t*n2+s*n+t]*rho(s)*eta(r)*Eta(p,t);
                            Outr(p,s) += 2.*Vi_ss[p*n3+s*n2+p*n+r]*g0[p*n3+s*n2+p*n+r]*Vi_ss[r*n3+t*n2+s*n+t]*Rho(t,s)*eta(r)*Eta(p,t);
                            Outr(p,t) += -2.*Vi_ss[p*n3+s*n2+r*n+s]*g0[p*n3+s*n2+r*n+s]*Vi_ss[r*n3+t*n2+p*n+t]*Rho(p,s)*Eta(s,r)*Eta(r,t);
                            Outr(p,p) += 2.*Vi_ss[p*n3+s*n2+r*n+s]*g0[p*n3+s*n2+r*n+s]*Vi_ss[r*n3+t*n2+p*n+t]*Rho(t,s)*Eta(s,r)*Eta(r,t);
                            Outr(p,r) += 2.*Vi_ss[p*n3+s*n2+s*n+t]*g0[p*n3+s*n2+s*n+t]*Vi_ss[r*n3+t*n2+p*n+r]*Rho(p,s)*Eta(s,r)*eta(t);
                            Outr(p,p) += -2.*Vi[p*n3+s*n2+t*n+s]*g0[p*n3+s*n2+t*n+s]*Vi[t*n3+r*n2+p*n+r]*Rho(r,s)*Eta(s,r)*eta(t);
                            Outr(p,p) += -2.*Vi_ss[p*n3+s*n2+s*n+t]*g0[p*n3+s*n2+s*n+t]*Vi_ss[r*n3+t*n2+p*n+r]*Rho(r,s)*Eta(s,r)*eta(t);
                        }
                    }
                }
            }
            
#pragma omp parallel for
            for(int q=0; q<n; ++q)
            {
                for(int r=0; r<n; ++r)
                {
                    for(int s=0; s<n; ++s)
                    {
                        for(int t=0; t<n; ++t)
                        {
                            Oute2(q,q) += -1.*Vi_ss[s*n3+t*n2+q*n+r]*g0[s*n3+t*n2+q*n+r]*Vi_ss[q*n3+r*n2+s*n+t]*Rho(t,s)*Rho(s,t)*eta(r);
                            Oute2(q,q) += 1.*Vi[s*n3+t*n2+q*n+r]*g0[s*n3+t*n2+q*n+r]*Vi[q*n3+r*n2+s*n+t]*rho(s)*rho(t)*eta(r);
                            Oute2(q,q) += 1.*Vi[t*n3+s*n2+q*n+r]*g0[t*n3+s*n2+q*n+r]*Vi[q*n3+r*n2+t*n+s]*rho(s)*rho(t)*eta(r);
                            Oute2(q,q) += 1.*Vi_ss[s*n3+t*n2+q*n+r]*g0[s*n3+t*n2+q*n+r]*Vi_ss[q*n3+r*n2+s*n+t]*rho(s)*rho(t)*eta(r);
                            Outr2(t,q) += -2.*Vi_ss[q*n3+r*n2+q*n+t]*g0[q*n3+r*n2+q*n+t]*Vi_ss[s*n3+t*n2+r*n+s]*Rho(t,s)*Eta(s,q)*eta(r);
                            Outr2(t,q) += -2.*Vi[q*n3+r*n2+q*n+s]*g0[q*n3+r*n2+q*n+s]*Vi[t*n3+s*n2+t*n+r]*rho(s)*Eta(t,q)*eta(r);
                            Outr2(t,q) += -2.*Vi_ss[q*n3+r*n2+q*n+s]*g0[q*n3+r*n2+q*n+s]*Vi_ss[s*n3+t*n2+r*n+t]*rho(s)*Eta(t,q)*eta(r);
                            Oute2(q,q) += -2.*Vi_ss[s*n3+t*n2+q*n+s]*g0[s*n3+t*n2+q*n+s]*Vi_ss[q*n3+r*n2+r*n+t]*Rho(t,s)*Rho(r,t)*Eta(s,r);
                            Oute2(q,q) += 2.*Vi[t*n3+s*n2+q*n+s]*g0[t*n3+s*n2+q*n+s]*Vi[q*n3+r*n2+t*n+r]*Rho(r,s)*rho(t)*Eta(s,r);
                            Oute2(q,q) += 2.*Vi_ss[s*n3+t*n2+q*n+s]*g0[s*n3+t*n2+q*n+s]*Vi_ss[q*n3+r*n2+r*n+t]*Rho(r,s)*rho(t)*Eta(s,r);
                            Outr2(t,q) += 2.*Vi_ss[q*n3+r*n2+q*n+t]*g0[q*n3+r*n2+q*n+t]*Vi_ss[s*n3+t*n2+r*n+s]*Rho(t,s)*Eta(r,q)*Eta(s,r);
                            Outr2(s,q) += -2.*Vi_ss[q*n3+r*n2+q*n+t]*g0[q*n3+r*n2+q*n+t]*Vi_ss[s*n3+t*n2+r*n+s]*rho(t)*Eta(r,q)*Eta(s,r);
                            Oute2(s,q) += -2.*Vi_ss[r*n3+t*n2+q*n+r]*g0[r*n3+t*n2+q*n+r]*Vi_ss[q*n3+s*n2+s*n+t]*Rho(s,r)*rho(t)*Eta(r,q);
                            Oute2(s,q) += 2.*Vi_ss[r*n3+t*n2+q*n+t]*g0[r*n3+t*n2+q*n+t]*Vi_ss[q*n3+s*n2+r*n+s]*Rho(s,r)*Rho(r,t)*Eta(t,q);
                            Oute2(s,q) += 2.*Vi_ss[q*n3+t*n2+q*n+s]*g0[q*n3+t*n2+q*n+s]*Vi_ss[r*n3+s*n2+r*n+t]*Rho(t,q)*Rho(r,t)*Eta(s,r);
                            Oute2(s,q) += -2.*Vi_ss[q*n3+t*n2+q*n+s]*g0[q*n3+t*n2+q*n+s]*Vi_ss[r*n3+s*n2+r*n+t]*Rho(r,q)*rho(t)*Eta(s,r);
                            Outr2(t,q) += -1.*Vi_ss[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi_ss[q*n3+t*n2+r*n+s]*Rho(t,q)*Eta(s,r)*Eta(r,s);
                            Outr2(q,q) += 1.*Vi_ss[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi_ss[q*n3+t*n2+r*n+s]*rho(t)*Eta(s,r)*Eta(r,s);
                            Outr2(t,q) += 1.*Vi_ss[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi_ss[q*n3+t*n2+r*n+s]*Rho(t,q)*eta(r)*eta(s);
                            Outr2(q,q) += -1.*Vi[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi[q*n3+t*n2+r*n+s]*rho(t)*eta(r)*eta(s);
                            Outr2(q,q) += -1.*Vi[s*n3+r*n2+q*n+t]*g0[s*n3+r*n2+q*n+t]*Vi[q*n3+t*n2+s*n+r]*rho(t)*eta(r)*eta(s);
                            Outr2(q,q) += -1.*Vi_ss[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi_ss[q*n3+t*n2+r*n+s]*rho(t)*eta(r)*eta(s);
                            Oute2(t,q) += 1.*Vi_ss[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi_ss[q*n3+t*n2+r*n+s]*Rho(s,r)*Rho(r,s)*Eta(t,q);
                            Oute2(t,q) += -1.*Vi_ss[r*n3+s*n2+q*n+t]*g0[r*n3+s*n2+q*n+t]*Vi_ss[q*n3+t*n2+r*n+s]*rho(r)*rho(s)*Eta(t,q);
                            Oute2(t,q) += 2.*Vi[q*n3+s*n2+q*n+r]*g0[q*n3+s*n2+q*n+r]*Vi[t*n3+r*n2+t*n+s]*Rho(t,q)*rho(s)*eta(r);
                            Oute2(t,q) += 2.*Vi_ss[q*n3+s*n2+q*n+r]*g0[q*n3+s*n2+q*n+r]*Vi_ss[r*n3+t*n2+s*n+t]*Rho(t,q)*rho(s)*eta(r);
                            Oute2(t,q) += -2.*Vi_ss[q*n3+s*n2+q*n+r]*g0[q*n3+s*n2+q*n+r]*Vi_ss[r*n3+t*n2+s*n+t]*Rho(s,q)*Rho(t,s)*eta(r);
                            Outr2(s,q) += -2.*Vi_ss[r*n3+t*n2+q*n+t]*g0[r*n3+t*n2+q*n+t]*Vi_ss[q*n3+s*n2+r*n+s]*Rho(t,q)*Eta(s,r)*Eta(r,t);
                            Outr2(q,q) += 2.*Vi_ss[r*n3+t*n2+q*n+t]*g0[r*n3+t*n2+q*n+t]*Vi_ss[q*n3+s*n2+r*n+s]*Rho(t,s)*Eta(s,r)*Eta(r,t);
                            Outr2(s,q) += 2.*Vi_ss[r*n3+t*n2+q*n+r]*g0[r*n3+t*n2+q*n+r]*Vi_ss[q*n3+s*n2+s*n+t]*Rho(r,q)*Eta(s,r)*eta(t);
                            Outr2(q,q) += -2.*Vi[t*n3+r*n2+q*n+r]*g0[t*n3+r*n2+q*n+r]*Vi[q*n3+s*n2+t*n+s]*Rho(r,s)*Eta(s,r)*eta(t);
                            Outr2(q,q) += -2.*Vi_ss[r*n3+t*n2+q*n+r]*g0[r*n3+t*n2+q*n+r]*Vi_ss[q*n3+s*n2+s*n+t]*Rho(r,s)*Eta(s,r)*eta(t);
                        }
                    }
                }
            }
            
#pragma omp parallel for
            for(int q=0; q<n; ++q)
            {
                for(int r=0; r<n; ++r)
                {
                    for(int s=0; s<n; ++s)
                    {
                        for(int p=0; p<n; ++p)
                        {
                            Out(p,q) += -2.*Vi[p*n3+r*n2+p*n+s]*g0[p*n3+r*n2+p*n+s]*Vi[p*n3+s*n2+p*n+r]*rho(p)*rho(s)*Eta(p,q)*eta(r);
                            Out(p,q) += -2.*Vi[p*n3+r*n2+s*n+r]*g0[p*n3+r*n2+s*n+r]*Vi[s*n3+r*n2+p*n+r]*rho(r)*rho(s)*Eta(p,q)*eta(r);
                            Out(p,q) += -2.*Vi_ss[p*n3+r*n2+p*n+s]*g0[p*n3+r*n2+p*n+s]*Vi_ss[p*n3+s*n2+p*n+r]*rho(p)*rho(s)*Eta(p,q)*eta(r);
                            Out(p,q) += -2.*Vi_ss[p*n3+r*n2+p*n+s]*g0[p*n3+r*n2+p*n+s]*Vi_ss[p*n3+s*n2+p*n+r]*Rho(p,q)*rho(r)*Eta(s,p)*Eta(p,s);
                            Out(p,q) += -2.*Vi_ss[p*n3+r*n2+p*n+s]*g0[p*n3+r*n2+p*n+s]*Vi_ss[p*n3+s*n2+p*n+r]*Rho(s,p)*Rho(p,s)*Eta(r,q)*Eta(p,r);
                            Out(p,q) += -2.*Vi_ss[p*n3+r*n2+r*n+s]*g0[p*n3+r*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+r]*rho(r)*rho(s)*Eta(p,q)*eta(r);
                            Out(p,q) += -2.*Vi_ss[p*n3+r*n2+r*n+s]*g0[p*n3+r*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+r]*Rho(s,r)*Rho(r,s)*Eta(r,q)*Eta(p,r);
                            Out(p,q) += -2.*Vi_ss[p*n3+s*n2+p*n+r]*g0[p*n3+s*n2+p*n+r]*Vi_ss[p*n3+r*n2+p*n+s]*Rho(s,q)*Rho(p,s)*eta(p)*eta(r);
                            Out(p,q) += -2.*Vi_ss[q*n3+r*n2+q*n+s]*g0[q*n3+r*n2+q*n+s]*Vi_ss[q*n3+s*n2+q*n+r]*Rho(s,q)*Rho(p,s)*eta(q)*eta(r);
                            Out(p,q) += -2.*Vi[q*n3+s*n2+q*n+r]*g0[q*n3+s*n2+q*n+r]*Vi[q*n3+r*n2+q*n+s]*rho(q)*rho(s)*Eta(p,q)*eta(r);
                            Out(p,q) += -2.*Vi_ss[q*n3+s*n2+q*n+r]*g0[q*n3+s*n2+q*n+r]*Vi_ss[q*n3+r*n2+q*n+s]*Rho(p,q)*rho(r)*Eta(s,q)*Eta(q,s);
                            Out(p,q) += -2.*Vi_ss[q*n3+s*n2+q*n+r]*g0[q*n3+s*n2+q*n+r]*Vi_ss[q*n3+r*n2+q*n+s]*rho(q)*rho(s)*Eta(p,q)*eta(r);
                            Out(p,q) += -2.*Vi_ss[q*n3+s*n2+q*n+r]*g0[q*n3+s*n2+q*n+r]*Vi_ss[q*n3+r*n2+q*n+s]*Rho(s,q)*Rho(q,s)*Eta(r,q)*Eta(p,r);
                            Out(p,q) += -2.*Vi_ss[r*n3+s*n2+q*n+r]*g0[r*n3+s*n2+q*n+r]*Vi_ss[q*n3+r*n2+r*n+s]*rho(r)*rho(s)*Eta(p,q)*eta(r);
                            Out(p,q) += -2.*Vi_ss[r*n3+s*n2+q*n+r]*g0[r*n3+s*n2+q*n+r]*Vi_ss[q*n3+r*n2+r*n+s]*Rho(s,r)*Rho(r,s)*Eta(r,q)*Eta(p,r);
                            Out(p,q) += -2.*Vi[s*n3+r*n2+q*n+r]*g0[s*n3+r*n2+q*n+r]*Vi[q*n3+r*n2+s*n+r]*rho(r)*rho(s)*Eta(p,q)*eta(r);
                            Out(p,q) += -1.*Vi_ss[p*n3+r*n2+r*n+s]*g0[p*n3+r*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+r]*Rho(p,q)*rho(r)*Eta(s,r)*Eta(r,s);
                            Out(p,q) += -1.*Vi_ss[p*n3+r*n2+r*n+s]*g0[p*n3+r*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+r]*Rho(r,q)*Rho(p,r)*eta(r)*eta(s);
                            Out(p,q) += -1.*Vi_ss[p*n3+s*n2+r*n+s]*g0[p*n3+s*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+s]*Rho(p,q)*rho(s)*Eta(s,r)*Eta(r,s);
                            Out(p,q) += -1.*Vi_ss[p*n3+s*n2+r*n+s]*g0[p*n3+s*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+s]*Rho(s,q)*Rho(p,s)*eta(r)*eta(s);
                            Out(p,q) += -1.*Vi_ss[r*n3+s*n2+q*n+r]*g0[r*n3+s*n2+q*n+r]*Vi_ss[q*n3+r*n2+r*n+s]*Rho(p,q)*rho(r)*Eta(s,r)*Eta(r,s);
                            Out(p,q) += -1.*Vi_ss[r*n3+s*n2+q*n+r]*g0[r*n3+s*n2+q*n+r]*Vi_ss[q*n3+r*n2+r*n+s]*Rho(r,q)*Rho(p,r)*eta(r)*eta(s);
                            Out(p,q) += -1.*Vi_ss[r*n3+s*n2+q*n+s]*g0[r*n3+s*n2+q*n+s]*Vi_ss[q*n3+s*n2+r*n+s]*Rho(p,q)*rho(s)*Eta(s,r)*Eta(r,s);
                            Out(p,q) += -1.*Vi_ss[r*n3+s*n2+q*n+s]*g0[r*n3+s*n2+q*n+s]*Vi_ss[q*n3+s*n2+r*n+s]*Rho(s,q)*Rho(p,s)*eta(r)*eta(s);
                            Out(p,q) += 1.*Vi[p*n3+r*n2+s*n+r]*g0[p*n3+r*n2+s*n+r]*Vi[s*n3+r*n2+p*n+r]*Rho(p,q)*rho(r)*eta(r)*eta(s);
                            Out(p,q) += 1.*Vi_ss[p*n3+r*n2+r*n+s]*g0[p*n3+r*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+r]*Rho(p,q)*rho(r)*eta(r)*eta(s);
                            Out(p,q) += 1.*Vi_ss[p*n3+r*n2+r*n+s]*g0[p*n3+r*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+r]*Rho(r,q)*Rho(p,r)*Eta(s,r)*Eta(r,s);
                            Out(p,q) += 1.*Vi[p*n3+s*n2+r*n+s]*g0[p*n3+s*n2+r*n+s]*Vi[r*n3+s*n2+p*n+s]*Rho(p,q)*rho(s)*eta(r)*eta(s);
                            Out(p,q) += 1.*Vi_ss[p*n3+s*n2+r*n+s]*g0[p*n3+s*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+s]*Rho(p,q)*rho(s)*eta(r)*eta(s);
                            Out(p,q) += 1.*Vi_ss[p*n3+s*n2+r*n+s]*g0[p*n3+s*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+s]*Rho(s,q)*Rho(p,s)*Eta(s,r)*Eta(r,s);
                            Out(p,q) += 1.*Vi[r*n3+s*n2+q*n+s]*g0[r*n3+s*n2+q*n+s]*Vi[q*n3+s*n2+r*n+s]*Rho(p,q)*rho(s)*eta(r)*eta(s);
                            Out(p,q) += 1.*Vi_ss[r*n3+s*n2+q*n+r]*g0[r*n3+s*n2+q*n+r]*Vi_ss[q*n3+r*n2+r*n+s]*Rho(p,q)*rho(r)*eta(r)*eta(s);
                            Out(p,q) += 1.*Vi_ss[r*n3+s*n2+q*n+r]*g0[r*n3+s*n2+q*n+r]*Vi_ss[q*n3+r*n2+r*n+s]*Rho(r,q)*Rho(p,r)*Eta(s,r)*Eta(r,s);
                            Out(p,q) += 1.*Vi_ss[r*n3+s*n2+q*n+s]*g0[r*n3+s*n2+q*n+s]*Vi_ss[q*n3+s*n2+r*n+s]*Rho(p,q)*rho(s)*eta(r)*eta(s);
                            Out(p,q) += 1.*Vi_ss[r*n3+s*n2+q*n+s]*g0[r*n3+s*n2+q*n+s]*Vi_ss[q*n3+s*n2+r*n+s]*Rho(s,q)*Rho(p,s)*Eta(s,r)*Eta(r,s);
                            Out(p,q) += 1.*Vi[s*n3+r*n2+q*n+r]*g0[s*n3+r*n2+q*n+r]*Vi[q*n3+r*n2+s*n+r]*Rho(p,q)*rho(r)*eta(r)*eta(s);
                            Out(p,q) += 2.*Vi_ss[p*n3+r*n2+p*n+s]*g0[p*n3+r*n2+p*n+s]*Vi_ss[p*n3+s*n2+p*n+r]*Rho(s,p)*Rho(p,s)*Eta(p,q)*eta(r);
                            Out(p,q) += 2.*Vi_ss[p*n3+r*n2+r*n+s]*g0[p*n3+r*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+r]*Rho(s,r)*Rho(r,s)*Eta(p,q)*eta(r);
                            Out(p,q) += 2.*Vi[p*n3+s*n2+p*n+r]*g0[p*n3+s*n2+p*n+r]*Vi[p*n3+r*n2+p*n+s]*Rho(p,q)*rho(s)*eta(p)*eta(r);
                            Out(p,q) += 2.*Vi_ss[p*n3+s*n2+p*n+r]*g0[p*n3+s*n2+p*n+r]*Vi_ss[p*n3+r*n2+p*n+s]*rho(p)*rho(r)*Eta(s,q)*Eta(p,s);
                            Out(p,q) += 2.*Vi_ss[p*n3+s*n2+p*n+r]*g0[p*n3+s*n2+p*n+r]*Vi_ss[p*n3+r*n2+p*n+s]*Rho(p,q)*rho(s)*eta(p)*eta(r);
                            Out(p,q) += 2.*Vi_ss[p*n3+s*n2+p*n+r]*g0[p*n3+s*n2+p*n+r]*Vi_ss[p*n3+r*n2+p*n+s]*Rho(s,q)*Rho(p,s)*Eta(r,p)*Eta(p,r);
                            Out(p,q) += 2.*Vi_ss[p*n3+s*n2+r*n+s]*g0[p*n3+s*n2+r*n+s]*Vi_ss[r*n3+s*n2+p*n+s]*rho(r)*rho(s)*Eta(s,q)*Eta(p,s);
                            Out(p,q) += 2.*Vi[q*n3+r*n2+q*n+s]*g0[q*n3+r*n2+q*n+s]*Vi[q*n3+s*n2+q*n+r]*Rho(p,q)*rho(s)*eta(q)*eta(r);
                            Out(p,q) += 2.*Vi_ss[q*n3+r*n2+q*n+s]*g0[q*n3+r*n2+q*n+s]*Vi_ss[q*n3+s*n2+q*n+r]*Rho(p,q)*rho(s)*eta(q)*eta(r);
                            Out(p,q) += 2.*Vi_ss[q*n3+r*n2+q*n+s]*g0[q*n3+r*n2+q*n+s]*Vi_ss[q*n3+s*n2+q*n+r]*rho(q)*rho(r)*Eta(s,q)*Eta(p,s);
                            Out(p,q) += 2.*Vi_ss[q*n3+r*n2+q*n+s]*g0[q*n3+r*n2+q*n+s]*Vi_ss[q*n3+s*n2+q*n+r]*Rho(s,q)*Rho(p,s)*Eta(r,q)*Eta(q,r);
                            Out(p,q) += 2.*Vi_ss[q*n3+s*n2+q*n+r]*g0[q*n3+s*n2+q*n+r]*Vi_ss[q*n3+r*n2+q*n+s]*Rho(s,q)*Rho(q,s)*Eta(p,q)*eta(r);
                            Out(p,q) += 2.*Vi_ss[r*n3+s*n2+q*n+r]*g0[r*n3+s*n2+q*n+r]*Vi_ss[q*n3+r*n2+r*n+s]*Rho(s,r)*Rho(r,s)*Eta(p,q)*eta(r);
                            Out(p,q) += 2.*Vi_ss[r*n3+s*n2+q*n+s]*g0[r*n3+s*n2+q*n+s]*Vi_ss[q*n3+s*n2+r*n+s]*rho(r)*rho(s)*Eta(s,q)*Eta(p,s);
                        }
                    }
                }
            }
            
            Out += Oute*Eta;
            Out += Outr*Rho;
            Out += Eta*Oute2;
            Out += Rho*Outr2;
            
            RhoDot_ += 0.5*(Out+Out.t());
            
        }
        
        threading_policy::pop();
        return (trace(RhoDot_%Rho));
    }
    
    // Second born approximation.
    cx RhoDot2B(const cx_mat& Rho_, cx_mat& RhoDot_, const double& time, double epsilon=0.000000001, int g0flag=2)
    {
        cx_mat Rho(Rho_);
        
        for (int i=0; i<n; ++i)
        {
            if (real(Rho(i,i))<0)
                Rho(i,i)=0.0;
            if( abs(Rho(i,i))>1.0)
                Rho(i,i)=1.0;
        }
        
#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
        
#pragma omp declare reduction( + : cx_vec : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
#endif
        
        if (params["RIFock"]!=0.0)
            return 0.0;
        
        threading_policy::enable_omp_only();
        
        cx_mat Eta(Rho); Eta.eye();
        Eta -= Rho;
        
        cx_vec rho(Rho.diag());
        cx_vec eta(Eta.diag());
        
        cx_mat d2 = Delta2();
        cx_mat ed2 = exp(-j*d2*time);
        const cx* Vi = IntCur.memptr();
        
        if(1)
        {
            //cx_mat G0(n2,n2); cx* g0 = G0.memptr();
            cx_mat Out(RhoDot_); Out.zeros();
            cx_mat Tmp1(n2,n2); Tmp1.zeros();
            cx_mat Tmp2(n2,n2); Tmp2.zeros();
            cx_mat Tmp1p(n2,n2); Tmp1p.zeros();
            cx_mat Tmp2p(n2,n2); Tmp2p.zeros();
            
#pragma omp parallel for
            for(int m=0; m<n; ++m)
                for(int q=0; q<n; ++q)
                    for(int s=0; s<n; ++s)
                        for(int j=0; j<n; ++j)
                            for(int k=0; k<n; ++k)
                            {
                                Tmp1[m*n3+q*n2+s*n+k] += (2.0*Vi[m*n3+q*n2+s*n+j]-Vi[m*n3+q*n2+j*n+s])*Eta(j,k);
                                Tmp2[m*n3+q*n2+s*n+k] += (2.0*Vi[m*n3+q*n2+s*n+j]-Vi[m*n3+q*n2+j*n+s])*Rho(j,k);
                            }
            
#pragma omp parallel for
            for(int m=0; m<n; ++m)
                for(int q=0; q<n; ++q)
                    for(int s=0; s<n; ++s)
                        for(int k=0; k<n; ++k)
                            for(int r=0; r<n; ++r)
                            {
                                Tmp1p[m*n3+q*n2+r*n+k] += Tmp1[m*n3+q*n2+s*n+k]*Eta(s,r);
                                Tmp2p[m*n3+q*n2+r*n+k] += Tmp2[m*n3+q*n2+s*n+k]*Rho(s,r);
                            }
            Tmp1.zeros(); Tmp2.zeros();
            
#pragma omp parallel for
            for(int m=0; m<n; ++m)
                for(int p=0; p<n; ++p)
                    for(int q=0; q<n; ++q)
                        for(int k=0; k<n; ++k)
                            for(int r=0; r<n; ++r)
                            {
                                Tmp1[m*n3+p*n2+r*n+k] += Tmp1p[m*n3+q*n2+r*n+k]*Rho(p,q);
                                Tmp2[m*n3+p*n2+r*n+k] += Tmp2p[m*n3+q*n2+r*n+k]*Eta(p,q);
                            }
            Tmp1p.zeros(); Tmp2p.zeros();
            
#pragma omp parallel for
            for(int nn=0; nn<n; ++nn)
                for(int m=0; m<n; ++m)
                    for(int p=0; p<n; ++p)
                        for(int k=0; k<n; ++k)
                                for(int r=0; r<n; ++r)
                                {
                                    Tmp1p[nn*n3+p*n2+r*n+k] += Tmp1[m*n3+p*n2+r*n+k]*Rho(nn,m);
                                    Tmp2p[nn*n3+p*n2+r*n+k] += Tmp2[m*n3+p*n2+r*n+k]*Eta(nn,m);
                                }
           
            
                                
#pragma omp parallel for
            for(int i=0; i<n; ++i)
                for(int nn=0; nn<n; ++nn)
                    for(int p=0; p<n; ++p)
                        for(int k=0; k<n; ++k)
                            for(int r=0; r<n; ++r)
                                {
                                    Out(i,k) += Vi[i*n3+r*n2+p*n+nn]*Tmp1p[nn*n3+p*n2+r*n+k]/(eigs(k)+eigs(r)-eigs(nn)-eigs(p)+pow(10.0,-9.0));
                                    Out(i,k) += Vi[i*n3+r*n2+p*n+nn]*Tmp2p[nn*n3+p*n2+r*n+k]/(eigs(k)+eigs(r)-eigs(nn)-eigs(p)+pow(10.0,-9.0));
                                }
            
            RhoDot_ += 0.5*(Out+Out.t());
            
        }
        
        threading_policy::pop();
        return (trace(RhoDot_%Rho));
    }
    
};

#endif
