//
//  Created by Kevin Koh on 10/25/16.
//  routine called by genscfmansimple.c
//

#ifndef TDCIh
#define TDCIh


#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <armadillo>
#include <string>

#include "TCL_prop.h"
#include "TCL_polematrices.h"
#include "TCL_ActiveSpace.h"

#define eVPerAu 27.2113
#define FsPerAu 0.0241888

using namespace arma;

class TDCI: public TDSCF
{
public:
  
    cx_double c0;
    cx_double j;
    cx_mat cia;
    int n,n2,n3,n4,no,nv,nact;
    double thresh,Emp2,Ehf,nuclear_energy,tnow;
    
    cube BpqR;
    cx_mat IntCur, VCur,V,F,rho0;
    mat X,H,RSinv;
    vec eigs;
    
    //Parameters
    bool B_Built;
    bool InCore;
    bool IsOn;
    
    // class
    ASParameters* asp;
    // to be deleted
    cx_mat mu;
    
    
    TDCI(rks* the_scf_,const std::map<std::string,double> Params_ = std::map<std::string,double>(),TDSCF_Temporaries* Scratch_ = NULL): TDSCF(the_scf_, Params_, Scratch_), H(the_scf->H), X(the_scf->X), B_Built(false), thresh(pow(10.0,-11.0)), rho0(1,1), j(0.0,1.0)
    {
        cout << endl;
        cout << "===================================" << endl;
        cout << "|  Realtime TDCI module          |" << endl;
        cout << "===================================" << endl;
        cout << "| J. Parkhill, T. Nguyen          |" << endl;
        cout << "| K. Koh, J. Herr,  K. Yao        |" << endl;
        cout << "===================================" << endl;
        
        params = ReadParameters(Params_);
        
        
        // Prop part
        n = n_mo;
        cout << "n_ao:" << n_ao << " n_mo: " << n_mo << " n_e/2: " << n_occ+1 << endl;
        V.resize(n,n); V.eye();
        
        f_diagb.resize(n_mo);
        mu.resize(n,n);
        cx_mat mu_xb(n_ao,n_ao), mu_yb(n_ao,n_ao), mu_zb(n_ao,n_ao);
        {
            cx_mat Rhob(n,n); Rhob.zeros();
            InitializeLiouvillian(f_diagb,V,mu_xb,mu_yb,mu_zb);
            f_diag = f_diagb;
            C = the_scf->X*V;
        }
        
        if (ApplyNoOscField)
            Mus = new StaticField(mu_xb, mu_yb, mu_zb, FieldVector, FieldAmplitude, Tau, tOn);
        else
            Mus = new OpticalField(mu_xb, mu_yb, mu_zb, FieldVector, ApplyImpulse, ApplyCw, FieldAmplitude, FieldFreq, Tau, tOn, ApplyImpulse2, FieldAmplitude2, FieldFreq2, Tau2, tOn2);
        
        
        firstOrb = 0;
        lastOrb = n-1;
        cx_mat CMAT;
        CMAT = the_scf->X*V;
        C = the_scf->X*V;
        CMAT=C.submat(0,firstOrb,n_ao-1,lastOrb);
        f_diag.st().print("Initial Eigenvalues:");
        cout << "Excitation Gap: " << (f_diag(n_e)-f_diag(n_e-1)) << " Eh : " << (f_diag(n_e)-f_diag(n_e-1))*27.2113 << "eV" << endl;
        cout << "Condition of Fock Matrix: " << 1.0/(f_diag(n-1)-f_diag(0)) << " timestep (should be smaller) " << dt << endl;
        Mus->update(CMAT);
        
        
        if (!Restart)
        {
            Pol.resize(5000,npol_col);
            Pol.zeros();
        }
        else
            ReadRestart();
        
        if (SavePopulations)
        {
            Pops = mat(100,n+1);
            Pops.zeros();
        }
        
        if (SaveFockEnergies)
        {
            FockEs = mat(100,n+1);
            FockEs.zeros();
        }
        
        
        // EE2 part start
        threading_policy::enable_omp_only();
        int LenV = megtot();
        int IBASIS = rem_read(REM_IBASIS);
        int stat = STAT_NBASIS;
        int Job;
        int IBasisAux = rem_read(REM_AUX_BASIS);
        nuclear_energy = the_scf->nuclear_energy();
        
        ftnshlsstats(&n_aux,&stat, &IBasisAux);

        {
            n=n_mo;
            n2=n_mo*n_mo;
            n3=n2*n_mo;
            n4=n3*n_mo;
            
            no=n_occ;
            nv=n_mo-n_occ;
        }

        t=0; loop=0; loop1 = 0; loop2=0; loop3=0;
        
        // Preparation
        // Eigval, Vi, mu, should be prepared
        {
            
            
            InCore = (n4*sizeof(cx_double)/(1024*1024) < 10.0);
            if(!params["RIFock"])
                IntCur.resize(n2,n2);
            
            BpqR.resize(n_mo,n_mo,n_aux);
            RSinv.resize(n_aux,n_aux);
            {
                int tmp1 = rem_read(REM_LEVCOR);
                rem_write(103, REM_LEVCOR);
                trnjob(&Job);
                rem_write(tmp1, REM_LEVCOR);
                
                threading_policy::enable_blas_only();
                ISqrtPQ();
                threading_policy::pop();
                
                FileMan(FM_OPEN_RW, FILE_2C2E_INVSQ_INTS, 0, 0, 0, 0, 0);
                LongFileMan(FM_READ, FILE_2C2E_INVSQ_INTS, FM_DP, n_aux*n_aux, 0, FM_BEG, RSinv.memptr(), 0);
                
                FileMan(FM_CLOSE, FILE_2C2E_INVSQ_INTS, 0, 0, 0, 0, 0);
            }
            
            
            BuildB();
            
            Rho.resize(n,n); NewRho.resize(n,n);
            Rho.zeros(); NewRho.zeros();
            Rho.eye(); Rho.submat(n_e,n_e,n-1,n-1) *= 0.0;
            cx_mat Rho_;
            Rho_=Rho;
            cx_mat V_(V);
            
            if (!params["StartFromSetman"])
            {
                Rho_.eye();
                if (!params["ActiveSpace"])
                    Rho_.submat(n_occ,n_occ,n_mo-1,n_mo-1) *= 0.0;
                else
                    Rho_.submat(asp->ne,asp->ne,n-1,n-1) *= 0.0;
            }
            FockBuild(Rho_,V_);
            UpdateVi(V_);
            
            //Checking Hcis
            if (params["CIS"])
            {
                HCIS();
                cia.resize(no,nv);
                c0 = 1;
                Mus->update(the_scf->X*V_);
                Mus->InitializeExpectation(MakeRho());
            }
            
            else if (params["CISD"])
            {
                HCISD();
                cia.resize(no,nv);
                c0 = 1;
                Mus->update(the_scf->X*V_);
                Mus->InitializeExpectation(MakeRhoCISD());
            }
            
<<<<<<< HEAD
            Mus->update(the_scf->X*V_);
            Mus->InitializeExpectation(MakeRho());
=======
>>>>>>> ec3151a0ff590f58c8eb3ec65e153cdbf72073b9
        }
        
        cout << "All parameters prepared" << endl;
        
        
        // propagate with different options
        Prop();
        
        
    }
    
    
    void HCIS()
    {
        cx_double cc0;
        cx_mat ccia;
        cx_mat Hcis(1+no*nv,1+no*nv);
        
        cia.resize(no,nv);
        ccia.resize(no,nv);
        cc0 = 0;
        ccia.zeros();
        c0 = j * 1;
        
        CISDOT(cia,c0,ccia,cc0,0);
        
        Hcis(0,0) = cc0;
        for (int i = 0; i < no*nv; i++)
            Hcis(0,i+1) = ccia(i);
        
        c0 = 0;
        for(int i = 0; i < no*nv; i++)
        {
            c0 = 0; cc0 = 0;
            cia.zeros();ccia.zeros();
            cia(i) = j * 1;
            
            CISDOT(cia,c0,ccia,cc0,0);
            Hcis(i+1,0) = cc0;
            for(int j = 0; j < no * nv; j++)
                Hcis(i+1,j+1) = ccia(j);
        }
        Hcis.print("Hcis:");
        vec hciseig = eig_sym(Hcis);
        hciseig.print();
        
        c0=0;
        cia.zeros();
    }
    
    void HCISD()
    {
    }
    
    cx_mat MakeRho() // Should take C0 and Cia as arguments returns the density matrix.
    {
        Rho.resize(n,n);
        Rho.zeros();
        Rho.eye(); Rho.submat(n_e,n_e,n-1,n-1) *= 0.0;
        cx_mat RhoHF(Rho);
        Rho *= conj(c0)*c0;

        Rho.submat(0,no,no-1,n-1) += 1/sqrt(2)*conj(c0)*cia;
        Rho.submat(no,0,n-1,no-1) += 1/sqrt(2)*conj(cia)*c0;
        
        cx_mat eRho(Rho); eRho.zeros();
        cx_mat hRho(Rho); hRho.zeros();
        

        for (int a = 0; a < nv; a++)
            for(int i = 0; i< no; i++)
                Rho += conj(cia(i,a))*cia(i,a)*RhoHF;
        
        for (int a = 0; a < nv; a++)
            for(int i = 0; i< no; i++)
                for(int ap = 0; ap < nv; ap++)
                {
                    Rho(ap+no,a+no) += 0.5*conj(cia(i,ap))*cia(i,a);
//                    eRho(a+no,ap+no) -= conj(cia(i,a))*cia(i,ap);
                }
        
        
        for (int a = 0; a < nv; a++)
            for(int i = 0; i< no; i++)
                    for(int ip = 0; ip < no; ip++)
                    {
                        Rho(i,ip) -= 0.5*cia(i,a)*conj(cia(ip,a));
//                        hRho(i,ip) -= cia(i,a)*conj(cia(ip,a));
                    }
        
//        cout << "Trace Rho: " << trace(Rho) << endl;
//        cout << "Trace eRho: " << trace(eRho) << endl;
//        cout << "Trace hRho: " << trace(hRho) << endl;
//        cout << "c0^2: " << conj(c0)*c0 << endl;
//        cout << "cia^2: " << sum(conj(cia)%cia) << endl;
        
        return Rho;
    }
        
    cx_mat MakeRhoCISD() // Should take C0 and Cia as arguments returns the density matrix.
    {
    }
        
    bool Prop()
    {
        if (MaxIter)
            if (loop>MaxIter)
            {
                cout << "Exiting Because MaxIter exceeded. SerialNumber"<< SerialNumber << " Iter " << loop << " of " << MaxIter << endl;
                return false;
            }
        
        double WallHrPerPs=0;
        
        //Pol size
        Pol.resize(5000,4);
        Pops.resize(5000,1+1+no*nv+2); // take +2 out after fix
        tnow = 0;

        cout << "First Energy gap: " << eigs(1) - eigs(0) << endl;
        cout << "dt: " << dt << endl;
        if (params["CIS"])
            cout << "---TDCI---" << endl;
        else if (params["CISD"])
            cout << "---TDCISD---" << endl;
        
        // Integration
        while(true)
        {
            
            // Timestep
            if (params["CIS"])
            {
                CISRK4step();
                tnow+=dt;
                cx_vec cpop(1+no*nv);
                for(int i = 0; i < 1+no*nv; i++)
                {
                    if(i ==0)
                        cpop(i) = c0 * conj(c0);
                    else
                        cpop(i) = cia(i-1) * conj(cia(i-1));
                }
                
                if (SavePopulations && (loop2%((int)params["SavePopulationsEvery"])==0))
                {
                    cx_mat Pop = MakeRho();
                    Pops(loop2,0)=tnow;
                    for(int ii = 0; ii < no*nv+1; ii++)
                        Pops(loop2,ii+1) = real(Pop(ii,ii));
                }
            }
            
            else
            {
<<<<<<< HEAD
                cx_mat Pop = MakeRho();
                Pops(loop1,0)=tnow;
                for(int ii = 0; ii < no*nv+1; ii++)
                    Pops(loop1,ii+1) = real(Pop(ii,ii));
                Pops(loop1,3) = real(Pop(1,0));//take these parts out after
                Pops(loop1,4) = imag(Pop(1,0));
                if (Pops.n_rows-loop1<100)
                    Pops.resize(Pops.n_rows+20000,Pops.n_cols);
                loop1 += 1;
=======
                CISDRK4step();
                tnow+=dt;
                cx_vec cpop(1+no*nv);
                for(int i = 0; i < 1+no*nv; i++)
                {
                    if(i ==0)
                        cpop(i) = c0 * conj(c0);
                    else
                        cpop(i) = cia(i-1) * conj(cia(i-1));
                }
>>>>>>> ec3151a0ff590f58c8eb3ec65e153cdbf72073b9
                
                if (SavePopulations && (loop2%((int)params["SavePopulationsEvery"])==0))
                {
                    cx_mat Pop = MakeRhoCISD();
                    Pops(loop2,0)=tnow;
                    for(int ii = 0; ii < no*nv+1; ii++)
                        Pops(loop2,ii+1) = real(Pop(ii,ii));
                }
            }
            
            if (loop%DipolesEvery==0 && DipolesEvery)
            {
                
                vec tmp = Mus->Expectation(MakeRho());
                Pol(loop2,0) = tnow; // in atomic units.
                Pol(loop2,1) = tmp(0);
                Pol(loop2,2) = tmp(1);
                Pol(loop2,3) = tmp(2);
                
                if (Pol.n_rows-loop2<100)
                    Pol.resize(Pol.n_rows+20000,Pol.n_cols);
                loop2 +=1;
            }
            
            if (SaveEvery)
                if (loop%SaveEvery==0)
                {
                    if (SaveDipoles)
                    {
                        // Also write the density && natural orbitals so they can be visualized.
                        Pol.save(logprefix+"Pol"+".csv", raw_ascii);
                        if (SavePopulations)
                            Pops.save(logprefix+"Pops"+".csv", raw_ascii);
                        if (SaveFockEnergies)
                            FockEs.save(logprefix+"FockEigs"+".csv", raw_ascii);
                        // Save the density matrix for possible restart.
                        if (params["CanRestart"])
                            Save();
                    }
                    if (WriteDensities > 0)
                        WriteEH(Rho, P0, V, t, loop3);
                }
            
            // Finish Iteration==========
            WallHrPerPs = ((time(0)-wallt0)/(60.0*60.0))/(t*FsPerAu/1000.0);
            double Elapsed = (int)(time(0)-wallt0)/60.0;
            double Remaining;
            if (Stutter)
                Remaining = ((double)( loop%(Stutter+1) )*(Elapsed/loop));
            else
                Remaining = ((double)( MaxIter-loop )*(Elapsed/loop));
            
            if(StatusEvery)
                if (loop%StatusEvery==0)
                {
                    cout << std::setprecision(6);
                    cout << "ITER: " << loop << " T: " << tnow*FsPerAu << "(fs)  dt " << dt*FsPerAu << endl;
<<<<<<< HEAD
=======
                    //real(cpop.t()).print("C Pop");
>>>>>>> ec3151a0ff590f58c8eb3ec65e153cdbf72073b9
                }
            
            loop +=1;
            
            if (MaxIter)
                if (loop>MaxIter)
                {
                    cout << "Exiting Because MaxIter exceeded. SerialNumber"<< SerialNumber << " Iter " << loop << " of " << MaxIter << endl;
                    Posify(Rho,n_e);
                    RhoM12 = Rho;
                    return false;
                }
            if ((Stutter > 0) && (loop%Stutter==0) && loop>0)
            {
                Posify(Rho,n_e);
                RhoM12 = Rho;
                return true;
            }

        }
        
    }
    
    void CISRK4step()
    {
        cx_mat dRho_(MakeRho());
        
        // rk4 step
        cx_mat k1(cia), k2(cia), k3(cia), k4(cia);
        cx_double k01, k02, k03, k04;
        k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
        k01 = 0; k02 = 0; k03 = 0; k04 = 0;
        
        CISDOT(cia, c0, k1, k01, tnow);
        CISDOT(cia+k1*dt/2.0, c0+k01*dt/2.0, k2, k02, tnow+dt/2.0);
        CISDOT(cia+k2*dt/2.0, c0+k02*dt/2.0, k3, k03, tnow+dt/2.0);
        CISDOT(cia+k3*dt, c0+k03*dt, k4, k04, tnow+dt);
        
        cia += dt/6.0 * (k1 + 2*k2 + 2*k3 + k4);
        c0 += dt/6.0 * (k01 + 2*k02 + 2*k03 + k04);
        
        double sum1 = real(c0*conj(c0) + accu(cia%conj(cia)));
        cia /= sqrt(sum1);
        c0 /= sqrt(sum1);
        
        dRho_ = MakeRho() - dRho_;
        cout << endl << tnow << endl;
        dRho_.print("dRho");
        cout << endl;
        
        //cia.print("Cia");
        //cout << c0 << endl;
        return;
    }
    
<<<<<<< HEAD
    void CISDOT(cx_mat CIA, cx_double C0, cx_mat &ciadot, cx_double &c0dot, double t)
=======
    void CISDRK4step()
    {
    }
    
    void CISDOT(cx_mat CIA, cx_double C0, cx_mat &ciadot, cx_double &c0dot, double tnow)
>>>>>>> ec3151a0ff590f58c8eb3ec65e153cdbf72073b9
    {
        
        const cx_double* Vi = IntCur.memptr();
        mu.zeros();
        Mus->ApplyField(mu,t,IsOn);

        // c0dot
        for (int a = 0; a < nv; a++)
            for(int i = 0; i< no; i++)
                c0dot += j * CIA(a, i) * mu(i,no+a);
        // first term of ciadot
        for (int a = 0; a < nv; a++)
            for(int i = 0; i< no; i++)
                ciadot(i,a) += - j * (eigs(no + a)-eigs(i)) * CIA(i,a);
        // second term of ciadot
        for (int a = 0; a < nv; a++)
            for(int i = 0; i< no; i++)
                for(int ap = 0; ap < nv; ap++)
                    for(int ip = 0; ip < no; ip++)
                        ciadot(i,a) += - j * CIA(ip,ap) * (2 * Vi[(a+no) * n3 + ip * n2 + i * n + (ap+no)] - Vi[(a+no) * n3 + ip * n2 + (ap+no) * n + i]);
        
        
        // last term of ciadot
        for (int a = 0; a < nv; a++)
            for(int i = 0; i< no; i++)
                ciadot(i,a) += j * C0 * mu(i,no+a);
        for (int a = 0; a < nv; a++)
            for(int i = 0; i< no; i++)
                for(int ap = 0; ap < nv; ap++)
                    ciadot(i,a) += j /sqrt(2)* CIA(i,ap) * mu(no+a,no+ap);
        
        for (int a = 0; a < nv; a++)
            for(int i = 0; i< no; i++)
                for(int ip = 0; ip < nv; ip++)
                    ciadot(i,a) += -j /sqrt(2)* CIA(ip,a) * mu(i,ip);
        
        return;
    }

/*    
    void CISDDOT(cx_mat CIA, cx_double C0, cx_mat &ciadot, cx_double &c0dot, double tnow)
    {
        mu.zeros();
        Mus->ApplyField(mu,tnow,IsOn);
        
        const cx* Vi = Vso.memptr();
        cx_mat bai(2*nv,2*no); bai.zeros();
        
        // Only singlets.
        for (int i=0; i<2*no; ++i)
            for (int a=0; a<2*nv; ++a)
            {
                if ( (i < no) && !(a<nv) || !(i < no) && (a<nv))
                    continue;
                bai(a,i) = v(no*(a%nv)+(i%no));
            }
        
        bai /= sqrt(dot(bai,bai));
        int tno = 2*no;
        int tnv = 2*nv;
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
        
        cx_mat u(4.0*nv*nv,4.0*no*no); u.zeros();
        cx_mat Rab(2.0*nv,2.0*nv); Rab.zeros();
        cx_mat Rij(2.0*no,2.0*no); Rij.zeros();
        cx_mat wai(2.0*nv,2.0*no); wai.zeros();
        cx_mat wai1(4.0*nv*no,4.0*nv*no); wai1.zeros();
        cx_mat T(4.0*nv*nv,4.0*no*no); T.zeros();
        cx_mat u2(4.0*nv*nv,4.0*no*no); u2.zeros();
        
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
        cout << "*Mp2:                     " << TestMp2 << endl;
        
        //Term1:
        
        // U1 = {{S,1},{{\[Delta],1},{i},{j}},{{bt,1},{b},{j}},{{Vt,2},{d,e},{b,m}},{{bs,1},{a},{i}},{{Vs,2},{d,e},{a,m}}}
        // U2 = {{S,-1},{{bt,1},{b},{j}},{{Vt,2},{d,e},{b,i}},{{bs,1},{a},{i}},{{Vs,2},{d,e},{a,j}}}
        // U3 = {{S,1},{{bt,1},{b},{j}},{{Vt,2},{d,a},{b,m}},{{bs,1},{a},{i}},{{Vs,2},{i,d},{j,m}}}
        // U4 = {{S,-1},{{bt,1},{b},{j}},{{Vt,2},{a,e},{b,m}},{{bs,1},{a},{i}},{{Vs,2},{i,e},{j,m}}}
        // U5 = U2
        // U6 = {{S,1},{{\[Delta],1},{i},{j}},{{bt,1},{b},{j}},{{Vt,2},{d,e},{b,l}},{{bs,1},{a},{i}},{{Vs,2},{d,e},{a,l}}}
        // U7 = {{S,-1},{{bt,1},{b},{j}},{{Vt,2},{d,a},{b,l}},{{bs,1},{a},{i}},{{Vs,2},{i,d},{l,j}}}
        // U8 = {{S,1},{{bt,1},{b},{j}},{{Vt,2},{a,e},{b,l}},{{bs,1},{a},{i}},{{Vs,2},{i,e},{l,j}}}
        // U9 = {{S,1},{{bt,1},{b},{j}},{{Vt,2},{j,d},{i,m}},{{bs,1},{a},{i}},{{Vs,2},{d,b},{a,m}}}
        // U10 = {{S,-1},{{bt,1},{b},{j}},{{Vt,2},{j,d},{l,i}},{{bs,1},{a},{i}},{{Vs,2},{d,b},{a,l}}}
        // U11 = {{S,1},{{\[Delta],1},{a},{b}},{{bt,1},{b},{j}},{{Vt,2},{j,d},{l,m}},{{bs,1},{a},{i}},{{Vs,2},{i,d},{l,m}}}
        // U12 = {{S,-1},{{bt,1},{b},{j}},{{Vt,2},{j,a},{l,m}},{{bs,1},{a},{i}},{{Vs,2},{i,b},{l,m}}}
        // U13 = {{S,-1},{{bt,1},{b},{j}},{{Vt,2},{j,e},{i,m}},{{bs,1},{a},{i}},{{Vs,2},{b,e},{a,m}}}
        // U14 = {{S,1},{{bt,1},{b},{j}},{{Vt,2},{j,e},{l,i}},{{bs,1},{a},{i}},{{Vs,2},{b,e},{a,l}}}
        // U15 = U12
        // U16 = {{S,1},{{\[Delta],1},{a},{b}},{{bt,1},{b},{j}},{{Vt,2},{j,e},{l,m}},{{bs,1},{a},{i}},{{Vs,2},{i,e},{l,m}}}
        
        
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

        
    }
*/

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Start of EE2
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
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
    
    
    void BuildB()
    {
        cout << "Building B..." << endl;
        if (true)
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
    }
    
    cx_mat FockBuild(cx_mat& Rho, cx_mat& V_, bool debug=false)
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
            // Fock build should be done in prev eigenbasis instead...
            for (int R=0; R<n_aux; ++R)
            {
                mat B=BpqR.slice(R);
                mat I1=real(Rhol%B); // Because Rho is hermitian and B is hermitian only the real part of this matrix should actually contribute to J
                J += 2.0*B*accu(I1); // J should be a Hermitian matrix.
                K -= B*(Rhot)*B;
            }
            F = 0.5*h + J + K;
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

    
    void UpdateVi(const cx_mat& V_)
    {
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
                B = V_.t()*BpqR.slice(R)*V_; // No need to xform 5 times/iteration. I could do this once for a 5x speedup.
                // Manually OMP the kron to avoid memory death.
#pragma omp parallel for schedule(guided)
                for(int p=0; p<n; ++p)
                {
                    for(int q=0; q<n; ++q)
                    {
                        cx_double A = B(p,q);
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
    
    
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // End of EE2
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
    
};


#endif
