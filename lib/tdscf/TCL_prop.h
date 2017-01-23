#ifndef TCLproph
#define TCLproph
#define HASEE2 1
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <armadillo>
#include <string>

#include "TCL.h"
#include "TCL_ActiveSpace.h"

#if HASEE2
#include "TCL_EE2.h"
#else 
#include "TCL_polematrices.h"
#endif

#include "math.h"
#include "rks.h"

using namespace arma;

#define eVPerAu 27.2113
#define FsPerAu 0.0241888

class TDSCF
{
public:
	int n_ao, n_mo;
	int n_occ;
    int n_aux;
	int n, n_e;
	int n_fock;
    int loop, loop1, loop2, loop3;
    int firstOrb, lastOrb;
    int npol_col;
    double Entropy,Energy,Ex,Ec; // Energy of core density.
    double t;
    long double wallt0;
    
    rks* the_scf;
    TCLMatrices* MyTCL;
#if HASEE2
	EE2* ee2;
#else 
    void* ee2;
#endif
    FieldMatrices* Mus;
    TDSCF_Temporaries* Scratch;
    std::map<std::string,double> params;
    
    // For TDSSCF
    vec f_diag;
    cx_mat C;
    cx_mat HJK; // For incfock.
    mat P0;
    vec f_diagb;
    cx_mat V0,V; // (lowdinXFock change of basis.)
    ASParameters asp;
    /*
     Moved to ASParameters Structure
    cx_mat asp.Vs,asp.Csub; // (lowdinXFock change of basis for a sub-matrix.)
    // All these quantities are in the lowdin (x-basis) --------
    cx_mat asp.P0cx;  // Initial core density matrix (x-basis)
    cx_mat asp.P_proj; // Projector onto 'core' initial density
    cx_mat asp.Q_proj; // Complement of 'core' initial density.
    cx_mat asp.Ps_proj; // Projector onto 'core' initial density (rectangular)
    cx_mat asp.Qs_proj; // Complement of 'core' initial density. (rectangular)
    */
    cx_mat sH,sHJK; // Small H && HJK matrices in case update fock is off.
	cx_mat old_fock; // For incremental fock build
	cx_mat old_p;
    cx_mat Rho,NewRho,RhoM12;
    mat Pol;
    cx_vec LastFockPopulations;
    vec LastFockEnergies;
    arma::mat Pops; arma::mat FockEs;
    
	// Parameters ---------
    std::string logprefix;
	bool RunDynamics;
	double dt;
	int MaxIter;
	double RateBuildThresh;
	bool ActiveSpace;
	double ActiveCutoff;
	bool UpdateFockMatrix;
	bool IncFock;
	int UpdateEvery;
	bool UpdateMarkovRates;
    int Stabilize;
	int InitialCondition; // Depreciate?
    bool Restart;
	bool ApplyImpulse;
    bool ApplyNoOscField; // You can apply a static field.
	bool ApplyCw;
	double FieldFreq;
	double FieldAmplitude;
	vec FieldVector;
    double Tau; // PulseLength
	double tOn; // cental time of gaussian pulse.
    // Relaxation Conditions ---------
    double Temp;
	bool TCLOn;
    //--- Correlated Dynamics theory.
    bool Corr;
    bool RIFock;
    bool Mp2NOGuess;
    //--- TDCI
    bool CIS;
    bool CISD;
    //--- Pump Probe.
    int Stutter;
    int StartFromSetman;
	bool ApplyImpulse2; // These are unused normally owing to TDDFT's nonstationarity.
	double FieldFreq2;
	double FieldAmplitude2;
	double Tau2;
	double tOn2;
	// Logging ---------
    int SerialNumber;
	int DipolesEvery;
	int StatusEvery;
	int SaveFockEvery;
	int Print;
	bool SaveDipoles;
	bool SavePopulations;
	bool SaveFockEnergies;
	int WriteDensities;
	int SaveEvery;
	int FourierEvery;
	double FourierZoom;
	//debuging
    double FSCFT;
    // Argument over-rides any defaults || params read from disk.
    std::map<std::string,double> ReadParameters(std::map<std::string,double> ORParams_ = std::map<std::string,double>())
	{
		FieldVector = vec(3); FieldVector.zeros();
        std::map<std::string,double> vals;
		if (!FileExists("TDSCF.prm"))
		{
			cout << "===============================================" << endl;
			cout << "Didn't find parameters file. Using defaults!!! " << endl;
			cout << "===============================================" << endl;
		}
		else
		{
			ifstream prms("TDSCF.prm");
			std::string key; double val;
			prms.seekg(0);
			if(prms.is_open())
			{
				while(!prms.eof())
				{
					prms >> key >> val;
					vals[key]=val;
				}
			}
			else
			{
				cout << "Couldn't open Parameters :( ... " << endl;
				throw 1;
			}
			prms.close();
        }
        
        
        ORParams_.insert(vals.begin(), vals.end()); // Passed parameters over-ride defaults && TDSCF.prm.
        std::swap(vals,ORParams_);

        Stutter = ((vals.find("Stutter") != vals.end())? vals.find("Stutter")->second : 0);
        SerialNumber = ((vals.find("SerialNumber") != vals.end())? vals.find("SerialNumber")->second : 0);
        StartFromSetman = ((vals.find("StartFromSetman") != vals.end())? vals.find("StartFromSetman")->second : 0);
        RunDynamics = ((vals.find("RunDynamics") != vals.end())? vals.find("RunDynamics")->second : true);
        Corr = ((vals.find("Corr") != vals.end())? vals.find("Corr")->second : false);
        RIFock = ((vals.find("RIFock") != vals.end())? vals.find("RIFock")->second : false);
<<<<<<< HEAD
	//TDCI
        CIS = ((vals.find("CIS") != vals.end())? vals.find("CIS")->second : false);
        FSCFT = ((vals.find("FSCFT") != vals.end())? vals.find("FSCFT")->second : 0);
        
=======
        CIS = ((vals.find("CIS") != vals.end())? vals.find("CIS")->second : false); // TD-CI
        CISD = ((vals.find("CISD") != vals.end())? vals.find("CISD")->second : false); // TD-CISD
>>>>>>> ec3151a0ff590f58c8eb3ec65e153cdbf72073b9
        if ((vals.find("GML") == vals.end()))
            vals["GML"] = 0.0;
        else
            vals["GML"] = vals.find("GML")->second;
        if ((vals.find("FD") == vals.end()))
            vals["FD"] = 0.0;
        else
            vals["FD"] = vals.find("FD")->second;
        if ((vals.find("DumpEE2") == vals.end()))
            vals["DumpEE2"] = 0.0;
        else
            vals["DumpEE2"] = vals.find("FD")->second;
        Mp2NOGuess = ((vals.find("Mp2NOGuess") != vals.end())? vals.find("Mp2NOGuess")->second : false);
        dt = ((vals.find("dt") != vals.end())? vals.find("dt")->second : 0.02);
        RateBuildThresh = ((vals.find("RateBuildThresh") != vals.end())? vals.find("RateBuildThresh")->second : 0.001);
        ActiveSpace = ((vals.find("ActiveSpace") != vals.end())? vals.find("ActiveSpace")->second : false);
        ActiveCutoff = ((vals.find("ActiveCutoff") != vals.end())? vals.find("ActiveCutoff")->second : 15.0/27.2113);
        asp.ActiveFirst = ((vals.find("ActiveFirst") != vals.end())? vals.find("ActiveFirst")->second : 0);
        asp.ActiveLast = ((vals.find("ActiveLast") != vals.end())? vals.find("ActiveLast")->second : 0);
        Restart = ((vals.find("Restart") != vals.end())? vals.find("Restart")->second : false);
        Stabilize = ((vals.find("Stabilize") != vals.end())? vals.find("Stabilize")->second : 2);
        UpdateFockMatrix = ((vals.find("UpdateFockMatrix") != vals.end())? vals.find("UpdateFockMatrix")->second : true);
        int UpdateFockMatrixPP = ((vals.find("UpdateFockMatrixPP") != vals.end())? vals.find("UpdateFockMatrixPP")->second : true);
        if (!UpdateFockMatrixPP)
        {
            vals["UpdateFockMatrix"] = false;
            UpdateFockMatrix = false;
        }
        UpdateEvery = ((vals.find("UpdateEvery") != vals.end())? vals.find("UpdateEvery")->second : 1);
        IncFock = ((vals.find("IncFock") != vals.end())? vals.find("IncFock")->second : false);
        UpdateMarkovRates = ((vals.find("UpdateMarkovRates") != vals.end())? vals.find("UpdateMarkovRates")->second : true);
        TCLOn = ((vals.find("TCLOn") != vals.end())? vals.find("TCLOn")->second : false);
        Temp = ((vals.find("Temp") != vals.end())? vals.find("Temp")->second : 300.0);
        if (vals.find("MaxIter") != vals.end())
        {
            MaxIter = vals.find("MaxIter")->second;
        }
        else
        {
             MaxIter =  vals["MaxIter"] = 10000;
        }
        InitialCondition = ((vals.find("InitialCondition") != vals.end())? vals.find("InitialCondition")->second : 0);
        ApplyImpulse = ((vals.find("ApplyImpulse") != vals.end())? vals.find("ApplyImpulse")->second : true);
        ApplyCw = ((vals.find("ApplyCw") != vals.end())? vals.find("ApplyCw")->second : false);
        FieldFreq = ((vals.find("FieldFreq") != vals.end())? vals.find("FieldFreq")->second : 1.1);
        Tau = ((vals.find("Tau") != vals.end())? vals.find("Tau")->second : 0.07);
        tOn = ((vals.find("tOn") != vals.end())? vals.find("tOn")->second : 7.0*Tau);
        FieldAmplitude = ((vals.find("FieldAmplitude") != vals.end())? vals.find("FieldAmplitude")->second : 0.001);
        FieldVector(0) = ((vals.find("ExDir") != vals.end())? vals.find("ExDir")->second : 0.0);
        FieldVector(1) = ((vals.find("EyDir") != vals.end())? vals.find("EyDir")->second : 1.0);
        FieldVector(2) = ((vals.find("EzDir") != vals.end())? vals.find("EzDir")->second : 0.0);
        ApplyImpulse2 = ((vals.find("ApplyImpulse2") != vals.end())? vals.find("ApplyImpulse2")->second : false);
        ApplyNoOscField = ((vals.find("ApplyNoOscField") != vals.end())? vals.find("ApplyNoOscField")->second : false);
        FieldFreq2 = ((vals.find("FieldFreq2") != vals.end())? vals.find("FieldFreq2")->second : 3.0/27.2113);
        FieldAmplitude2 = ((vals.find("FieldAmplitude2") != vals.end())? vals.find("FieldAmplitude2")->second : 0.001);
        Tau2 = ((vals.find("Tau2") != vals.end())? vals.find("Tau2")->second : 0.07);
        tOn2 = ((vals.find("tOn2") != vals.end())? vals.find("tOn2")->second :  tOn+10.0/0.024);
        StatusEvery = ((vals.find("StatusEvery") != vals.end())? vals.find("StatusEvery")->second : 15);
        Print = ((vals.find("Print") != vals.end())? vals.find("Print")->second : 0);
        SaveDipoles = ((vals.find("SaveDipoles") != vals.end())? vals.find("SaveDipoles")->second : true);
        SavePopulations = ((vals.find("SavePopulations") != vals.end())? vals.find("SavePopulations")->second : false);
        if ((vals.find("PrntPopulationType") == vals.end()))
            vals["PrntPopulationType"] = 0.0; // 0 = Fock orbital populations, 1 = Natural orbital populations.
        if ((vals.find("SavePopulationsEvery") == vals.end()))
            vals["SavePopulationsEvery"] = 100.0;
        if ((vals.find("TSample") == vals.end()))
            vals["TSample"] = 1000.0; // The interval between transient absorption jobs. (fs)
        if ((vals.find("TDecoh") == vals.end()))
            vals["TDecoh"] = 100.0; // The decoherence time (fs)
        if((vals.find("ExcitedFraction") == vals.end()))
            vals["ExcitedFraction"] = 1.0;
        SaveFockEnergies = ((vals.find("SaveFockEnergies") != vals.end())? vals.find("SaveFockEnergies")->second : 0);
        SaveFockEvery = ((vals.find("SaveFockEvery") != vals.end())? vals.find("SaveFockEvery")->second : 100);
        WriteDensities = ((vals.find("WriteDensities") != vals.end())? vals.find("WriteDensities")->second : 0);
        SaveEvery = ((vals.find("SaveEvery") != vals.end())? vals.find("SaveEvery")->second : 250);
        DipolesEvery = ((vals.find("DipolesEvery") != vals.end())? vals.find("DipolesEvery")->second : 2);
        FourierEvery = ((vals.find("FourierEvery") != vals.end())? vals.find("FourierEvery")->second : 5000);
        FourierZoom = ((vals.find("FourierZoom") != vals.end())? vals.find("FourierZoom")->second : 0.268*(DipolesEvery*dt));
        if(Stutter!=0 && MaxIter!=0 && Stutter > MaxIter)
            Stutter = MaxIter-1;
        if(SaveEvery!=0 && MaxIter!=0 && SaveEvery > MaxIter)
            SaveEvery = MaxIter-1;
        
        
        // Set up separate logging prefixes if there is more than one propagation going on.
        if (SerialNumber!=0)
        {
            char str[12];
            sprintf(str, "%i", SerialNumber);
            std::string suffix(str);
            logprefix=logprefix+std::string("S")+suffix+std::string("_");
        }
        
		cout << "CONDITIONS:-----------" << endl;
        cout << "RIFock: " << RIFock << endl;
        if (Corr)
        {
            cout << "Corr: " << Corr << endl;
            cout << "GML (Gell-Man Low): " << vals["GML"] << endl;
            cout << "Mp2NOGuess: " << Mp2NOGuess << endl;
        }
        cout << "TCLOn: " << TCLOn << endl;
        cout << "Temp: " << Temp << endl;
        if (TCLOn)
        {
            cout << "UpdateMarkovRates: " << UpdateMarkovRates << endl;
            cout << "RateBuildThresh: " << RateBuildThresh << endl;
        }
		cout << "InitialCondition: " << InitialCondition << endl;
		cout << "StartFromSetman: " << StartFromSetman << endl;
		cout << "MaxIter: " << MaxIter << endl;
		cout << "Restart: " << Restart << endl;
		cout << "FieldAmplitude: " << FieldAmplitude << endl;
		cout << "ApplyImpulse: " << ApplyImpulse << endl;
		cout << "ApplyCw: " << ApplyCw << endl;
		cout << "FieldFreq: " << FieldFreq << endl;
		cout << "Tau: " << Tau << endl;
		cout << "tOn: " << tOn << endl;
		FieldVector.st().print("FieldVector");
        if (ApplyNoOscField)
        {
            cout << "ApplyNoOscField: " << ApplyNoOscField << endl;
            cout << "Tau: " << Tau << endl;
            cout << "tOn: " << tOn << endl;
        }
		if (ApplyImpulse2)
		{
			cout << "ApplyImpulse2: " << ApplyImpulse2 << endl;
			cout << "FieldAmplitude2: " << FieldAmplitude2 << endl;
			cout << "FieldFreq2: " << FieldFreq2 << endl;
			cout << "Tau2: " << Tau2 << endl;
			cout << "tOn2: " << tOn2 << endl;
		}
        
        cout << "THRESHOLDS: -----------" << endl;
		cout << "dt: " << dt << endl;
		cout << "ActiveSpace: " << ActiveSpace << endl;
        if (asp.ActiveLast!=0)
        {
            cout << "ActiveFirst: " << asp.ActiveFirst << endl;
            cout << "ActiveLast: " << asp.ActiveLast << endl;
        }
        else
            cout << "ActiveSpaceCutoff: " << ActiveCutoff << endl;
		cout << "UpdateFockMatrix: " << UpdateFockMatrix << endl;
        if (UpdateFockMatrix && UpdateEvery!=1)
            cout << "UpdateFockMatrix at (n) Steps: " << UpdateEvery << endl;
		cout << "IncFock : " << IncFock << endl;
		cout << "Stabilize: " << Stabilize << endl;
        
		cout << "LOGGING:--------------" << endl;
        cout << "SerialNumber: " << SerialNumber << endl;
        cout << "logprefix: " << logprefix << endl;
		cout << "Stutter: " << Stutter << endl;
        cout << "StatusEvery: " << StatusEvery << endl;
		cout << "DipolesEvery: " << DipolesEvery << endl;
		cout << "Print: " << Print << endl;
		cout << "SaveDipoles: " << SaveDipoles << endl;
		cout << "SaveEvery: " << SaveEvery << endl;
		cout << "WriteDensities: " << WriteDensities << endl;
		cout << "FourierEvery: " << FourierEvery << endl;
		cout << "FourierZoom: " << FourierZoom << endl;
		cout << "SavePopulations: " << SavePopulations << endl;
        cout << "SavePopulationsEvery: " << vals["SavePopulationsEvery"] << endl;
		cout << "SaveFockEnergies: " << SaveFockEnergies << endl;
		cout << "SaveFockEvery: " << SaveFockEvery << endl;
		cout << "======================" << endl << endl;
        return vals;
	}
	
    void SetupActiveSpace()
    {
        
        cout << "--------------------------" << endl;
        cout << "----Active Space Setup----" << endl;
        cout << "--------------------------" << endl;
        if (asp.ActiveFirst==0)
        {
            for (; firstOrb<n; ++firstOrb)
                if (abs(f_diagb(firstOrb)-f_diagb(n_occ-1))<ActiveCutoff)
                    break;
            lastOrb = firstOrb+1;
            for (; lastOrb<n; ++lastOrb)
                if (abs(f_diagb(lastOrb)-f_diagb(n_occ-1))>ActiveCutoff)
                    break;
            if (lastOrb < n_occ + 3)
                lastOrb = min(n,lastOrb+8);
            if (lastOrb == n)
                    lastOrb = n-1;
            cout << "Propagating within the sub-matrix: " << firstOrb << " to " << lastOrb << " (inclusive)"  << endl;
            cout << "Propagating within the sub-matrix: " << f_diagb(firstOrb) << " (eH) to " << f_diagb(lastOrb) << " eH (inclusive)"  << endl;
            cout << "Propagating within the sub-matrix: " << 27.2113*f_diagb(firstOrb) << " (eV) to " << 27.2113*f_diagb(lastOrb) << " eV (inclusive)"  << endl;
            asp.ActiveFirst = firstOrb;
            asp.ActiveLast = lastOrb;
        }
        else
        {
            firstOrb = asp.ActiveFirst;
            lastOrb = asp.ActiveLast;
        }
        
        int nact = lastOrb-firstOrb+1;
        cout << "Number of Active Space Orbitals: " << nact << endl;
        
        // Generate projectors these are in the Lowdin (X) basis.
        cx_mat P_(n,n); P_.eye();
        asp.P0cx=P_;
        asp.P0cx.submat(firstOrb,firstOrb,n-1,n-1) *= 0;
        //asp.P0cx is the core density matrix in the lowdin basis.
        asp.P0c = asp.P0cx;
        asp.P0cx = V*asp.P0cx*V.t(); // Lowdin basis core density matrix.
        asp.P0c.print("Original Density without population from first orbital to n in MO basis:");
        asp.P0cx.print("Density(Above) in Lowdin Basis(core density matrix):");
        P_.submat(firstOrb,firstOrb,lastOrb,lastOrb) *= 0.0;
        asp.P_proj = V*P_*V.t(); // Lowdin basis core inactive projector
        asp.Q_proj = eye<cx_mat>(n,n) - V*P_*V.t(); // Lowdin basis active projector
        P_.print("Original Density without population from first orbital to last orbital for active space in MO basis:");
        asp.P_proj.print("Density(Above) in Lowdin Basis(Inactive projector):");
        asp.Q_proj.print("Density in Lowdin Basis(Active projector):");
        
        // Make the rectangular versions too:
        asp.Ps_proj.zeros(); asp.Qs_proj.zeros();
        asp.Ps_proj.resize(n,n-nact); // projector onto inactive space (low X fock)
        int j=0;
        for (int i=0;i<n;++i)
        {
            if (i<firstOrb ||  i>lastOrb)
            {
                asp.Ps_proj.col(j) = V.col(i);
                ++j;
            }
        }
        asp.Qs_proj.resize(n,nact); // projector onto active space (low X fock)
        j=0;
        for (int i=0;i<n;++i)
        {
            if (i>=firstOrb && i<=lastOrb)
            {
                asp.Qs_proj.col(j) = V.col(i);
                ++j;
            }
        }
        cout << "N Core St: " << trace(real(asp.P_proj)) << " N Active St:" << trace(real(asp.Q_proj)) << endl;
        P_.eye(); P_.submat(the_scf->NOcc,the_scf->NOcc,n-1,n-1) *= 0.0; cx_mat Px = V*P_*V.t();
        cout << "Tr(Ps Core-Space)" << trace(asp.Ps_proj.t()*Px*asp.Ps_proj) << " Tr(Ps Active Space)" << trace(asp.Qs_proj.t()*Px*asp.Qs_proj) << endl;
        cout << " Tr(Pc Active Space) (should be zero) " << trace(asp.Qs_proj.t()*asp.P0cx*asp.Qs_proj) << endl;
        f_diag = f_diagb.subvec(firstOrb,lastOrb);
        n = lastOrb-firstOrb+1;
        n_e = n_occ - firstOrb;
        asp.ne = n_e;
        asp.Vs = asp.Qs_proj; // LAO->current eigenbasis.
        asp.Vs0 = asp.Qs_proj; // LAO->current eigenbasis.
        asp.Csub = the_scf->X*asp.Vs;
        old_fock.zeros();
        old_p.zeros();
        
        
        cout << "--------------------------" << endl;
        cout << "--------------------------" << endl;
        cout << "--------------------------" << endl;
    }
    
    void PlotFields()
    {
        cx_mat Field(n_mo,n_mo); Field.fill(0.0);
        double x=10.0; bool y = false;
        mat FieldR = real(Mus->mux);
        vec eigval; mat Cu;
        eig_sym(eigval,Cu,FieldR);//the_scf->AOS);
        String fileName  = String("./Perturbation.molden");
        FILE *fp = fopen(fileName.c_str(),"w"); //QOpen(fileName,"w");
        WriteMoldenATOMS_Repositioned(fp);
        WriteMoldenGTO(fp);
        vec dTs(n_mo); dTs.zeros();
        dTs(0)=99.0;
        WriteMoldenDiffDen(Cu.memptr(),Cu.memptr(),dTs.memptr(),dTs.memptr(),Cu.memptr(),Cu.memptr(),
                           the_scf->NOcc,the_scf->NOrb-the_scf->NOcc,the_scf->NOcc,the_scf->NOrb-the_scf->NOcc,fp);
        fclose(fp);
        
        if (ApplyImpulse) // Generate a power spectrum of the impulse for reference.
        {
            mat imp(10000,2);
            for (int i=0; i<10000; i+=1)
            {
                imp(i,0)=dt*(double)i;
                imp(i,1)=Mus->ImpulseAmp(dt*(double)i);
            }
            mat four=Fourier(imp,FourierZoom);
            four.save(logprefix+"FourierImpulse"+".csv", raw_ascii);
        }
        
    }
    
    void SetmanDensity()
    {
        int n_vir = n_mo - n_occ;
        int StartState = max(rem_read(REM_RTPPSTATE),0);
        cout << "Initializing from Linear Response State (starts from 1):" << StartState << endl;
        
        // To read out of the files written by setman.
        // Reading the RPA from setman.
        vec Xv(n_occ*n_vir),Yv(n_occ*n_vir);
        Xv.zeros(); Yv.zeros();
        
        bool RPA = true;
        
        FileMan(FM_READ,FILE_SET_TMP_RPA_X,FM_DP,n_vir*n_occ,0+n_vir*n_occ*(StartFromSetman-1),FM_BEG,Xv.memptr());
        if(RPA)
            FileMan(FM_READ,FILE_SET_TMP_RPA_Y,FM_DP,n_vir*n_occ,0+n_vir*n_occ*(StartFromSetman-1),FM_BEG,Yv.memptr());
        
        mat X(n_occ,n_vir),Y(n_occ,n_vir);
        cx_mat RhoRPA(n_mo,n_mo); RhoRPA.zeros();
        
        for (int a=0; a<n_vir; ++a)
            for (int i=0; i<n_occ; ++i)
            {
                X(i,a) = Xv(i*n_vir + a);
                Y(i,a) = Yv(i*n_vir + a);
            }
        
        if (n_mo<12 ||  Print > 1)
        {
            X.print("X");
            Y.print("Y");
        }
        
        for (int i=0; i<n_occ; ++i)
        {
            for (int j=0; j<n_occ; ++j)
            {
                for (int a=0; a<n_vir; ++a)
                {
                    RhoRPA(i,j) -= X(i, a)*X(j, a);
                    RhoRPA(i,j) += Y(i, a)*Y(j, a);
                }
            }
        }
        
        for (int b=0; b<n_vir; ++b)
        {
            for (int a=0; a<n_vir; ++a)
            {
                for (int i=0; i<n_occ; ++i)
                {
                    RhoRPA(a + n_occ,b+n_occ) += X(i,a)*X(i,b);
                    RhoRPA(a + n_occ,b+n_occ) -= Y(i,a)*Y(i,b);
                }
            }
        }
        
        // Add this to the orig density matrix in the active space.
        Rho += (params["ExcitedFraction"])*RhoRPA.submat(firstOrb,firstOrb,lastOrb,lastOrb);
        
        cout << "Constructed Particle-Hole Density" << endl;
        cout << "Tr(Rho)" << trace(Rho) << endl;
        cout << "Idempotency: " << accu(Rho-Rho*Rho) << endl;
        
        Rho.diag().st().print("Raw Populations:");
        
        cout << "Ensuring positivity." << endl;
        Posify(Rho,n_e);
        Rho.diag().st().print("Initial Populations:");
        
    }
    
    void ReadRestart()
    {
        if (!(FileExists((logprefix+"Rho_lastsave").c_str())))
        {
            cout << "Cannot Restart... missing data." << endl;
            Pol.resize(5000,npol_col);
            Pol.zeros(); // Polarization Energy, Trace, Gap, Entropy, HOMO-LUMO Coherence.
        }
        else
        {
            try{
                cout << "Reading old outputs to restart..." << endl;
                // Read in the old Pol.
                Pol.load(logprefix+"Pol.csv",raw_ascii);
                int i=2;
                for (; i<Pol.n_rows; ++i)
                    if (Pol(i,0) == 0.0)
                        break;
                if (i == Pol.n_rows-1)
                    cout << "Trouble Reading Pol File... " << endl;
                t = Pol(i-1,0);
                loop=i*DipolesEvery;
                loop2=i;
                loop3=int(loop/SaveEvery);
                cout << " *** Disabling save of fock energies && populations. *** " << endl;
                SavePopulations = false;
                SaveFockEnergies = false;
                cout << "Reading Active space Rho && starting from there." << endl;
                f_diag.load(logprefix+"f_diag_lastsave");
                Rho.load(logprefix+"Rho_lastsave");
                V.load(logprefix+"V_lastsave");
                Mus->load(logprefix+"Mus_lastsave_");
                asp.Vs.load(logprefix+"asp.Vs_lastsave");
                if (UpdateFockMatrix)
                {
                    HJK.load(logprefix+"HJK_lastsave");
                    RhoM12.load(logprefix+"RhoM12_lastsave");
                }
            }
            catch(...)
            {
                cout << "Restart -------- FAILURE ------- " << endl;
                throw;
            }
        }
    }
    
	void SplitLiouvillian(const arma::cx_mat& oldrho, arma::cx_mat& newrho, const double& tnow, bool& IsOn) const
	{
		if (TCLOn)
			MyTCL->ContractGammaRhoMThreads(newrho,oldrho);
	}
    
	void Split_RK4_Step(const arma::vec& f_diag, const arma::cx_mat& oldrho, arma::cx_mat& newrho, const double tnow, const double dt, bool& IsOn) const
	{
        if (!is_finite(oldrho) ||  !is_finite(f_diag))
        {
            cout << "Split_RK4_Step passed Garbage" << endl;
            throw 1;
        }
        
		arma::cx_mat f_halfstep(newrho); f_halfstep.eye();
		arma::cx_mat RhoHalfStepped(newrho); RhoHalfStepped.zeros();
        
        cx_vec f_diagc(f_diag.n_elem); f_diagc.set_real(f_diag);
        arma::cx_mat F = diagmat(f_diagc);
		Mus->ApplyField(F,tnow,IsOn);
        
        // Exponential propagator for Mu.
        vec eigval;
        cx_mat Cu;
        eig_sym(eigval,Cu,F);
        cx_mat Ud = exp(eigval*std::complex<double>(0.0,-0.5)*dt);
        cx_mat U = Cu*diagmat(Ud)*Cu.t();
		RhoHalfStepped = U*oldrho*U.t();
        
        if (!is_finite(newrho))
        {
            cout << "Garbage output in RhoHalfStepped" << endl;
            F.print("F");
            oldrho.print("oldrho");
            throw 1;
        }
        
		arma::cx_mat k1(newrho),k2(newrho),k3(newrho),k4(newrho),v2(newrho),v3(newrho),v4(newrho);
		k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
		v2.zeros(); v3.zeros(); v4.zeros();
		
		SplitLiouvillian( RhoHalfStepped, k1,tnow,IsOn);
		v2 = (dt/2.0) * k1;
		v2 += RhoHalfStepped;
		SplitLiouvillian( v2, k2,tnow+(dt/2.0),IsOn);
		v3 = (dt/2.0) * k2;
		v3 += RhoHalfStepped;
		SplitLiouvillian( v3, k3,tnow+(dt/2.0),IsOn);
		v4 = (dt) * k3;
		v4 += RhoHalfStepped;
		SplitLiouvillian( v4, k4,tnow+dt,IsOn);
		newrho = RhoHalfStepped;
		newrho += dt*(1.0/6.0)*k1;
		newrho += dt*(2.0/6.0)*k2;
		newrho += dt*(2.0/6.0)*k3;
		newrho += dt*(1.0/6.0)*k4;
        newrho = U*newrho*U.t();
    }
    
	void Split_RK4_Step_MMUT(const arma::vec& eigval, const arma::cx_mat& Cu , const arma::cx_mat& oldrho, arma::cx_mat& newrho, const double tnow, const double dt, bool& IsOn) const
	{
        cx_mat Ud = exp(eigval*std::complex<double>(0.0,-0.5)*dt);
        cx_mat U = Cu*diagmat(Ud)*Cu.t();
		cx_mat RhoHalfStepped = U*oldrho*U.t();
        
		arma::cx_mat k1(newrho),k2(newrho),k3(newrho),k4(newrho),v2(newrho),v3(newrho),v4(newrho);
		k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
		v2.zeros(); v3.zeros(); v4.zeros();
		
		SplitLiouvillian( RhoHalfStepped, k1,tnow,IsOn);
		v2 = (dt/2.0) * k1;
		v2 += RhoHalfStepped;
		SplitLiouvillian(  v2, k2,tnow+(dt/2.0),IsOn);
		v3 = (dt/2.0) * k2;
		v3 += RhoHalfStepped;
		SplitLiouvillian(  v3, k3,tnow+(dt/2.0),IsOn);
		v4 = (dt) * k3;
		v4 += RhoHalfStepped;
		SplitLiouvillian(  v4, k4,tnow+dt,IsOn);
		newrho = RhoHalfStepped;
		newrho += dt*(1.0/6.0)*k1;
		newrho += dt*(2.0/6.0)*k2;
		newrho += dt*(2.0/6.0)*k3;
		newrho += dt*(1.0/6.0)*k4;
        newrho = U*newrho*U.t();
    }
	
    void MMUTStep(arma::vec& f_diag, arma::cx_mat& HJK, arma::cx_mat& V, arma::cx_mat& Rho_, arma::cx_mat& RhoM12_, const double tnow , const double dt,bool& IsOn)
    {
        cx_mat CPrime(Rho_);
        // Fock Rebuild Rho at this time.
        CPrime = UpdateLiouvillian(f_diag,HJK,Rho_); // This rotates Rho, we must also rotate RhoM12!!!
        RhoM12_ = CPrime.t()*RhoM12_*CPrime;
        if (Stabilize)
            Posify(RhoM12_, n_e);
        
        // Make the ingredients for U.
        arma::cx_mat F(Rho_); F.zeros(); F.set_real(diagmat(f_diag));
		Mus->ApplyField(F,tnow,IsOn);
        
        // Exponential propagator for Mu.
        vec eigval; cx_mat Cu;
        eig_sym(eigval,Cu,F);
        
        // Full step RhoM12 to make new RhoM12.
        cx_mat NewRhoM12(Rho_);
        cx_mat NewRho(Rho_);
        Split_RK4_Step_MMUT(eigval, Cu, RhoM12_, NewRhoM12, tnow, dt, IsOn);
        // Half step that to make the new Rho.
        Split_RK4_Step_MMUT(eigval, Cu, NewRhoM12, NewRho, tnow, dt/2.0, IsOn);
        Rho_ = 0.5*(NewRho+NewRho.t());
        RhoM12_ = 0.5*(NewRhoM12+NewRhoM12.t());
    }
    
    
	void InitializeLiouvillian(vec& f_diag_, cx_mat& V, cx_mat& mu_x_, cx_mat& mu_y_, cx_mat& mu_z_)
	{
		int n = the_scf->NOrb;
		cx_mat Rhob(n,n); Rhob.zeros();
		cout << "Initializing the Liouvillian for " << n << " orbitals " << endl;
		the_scf->UpdateLiouvillian(Ex,Ec,Scratch, f_diag_,V,mu_x_,mu_y_,mu_z_,Rhob,old_fock,old_p,false);
	}
	
	// arg is in current Eigenbasis && mapped to Lowdin.
	cx_mat FullRho(const cx_mat& arg) const
	{
        if (ActiveSpace)
        {
            // Enlarge the density into the whole x space.
            cx_mat tmp = (asp.Vs*arg*asp.Vs.t());
            if (Print)
                cout << "FullRho: " << trace(tmp)<< trace(asp.P0cx+tmp) << endl;
            return (asp.P0cx+tmp);
        }
        else
            return (V*arg*V.t());
	}
    
    // Sync State members with another TDSCF object.
    // Ie: after this routine the propagations are in the same state.
    // Obviously between the Active space, incremental builds, etc. the number of state
    // variables has gotten out of control, && needs to be objectified.
    void Sync(const TDSCF* other)
    {
        cout << "P: " << SerialNumber << " Syncing from " << other->SerialNumber << endl;
        Mus->Sync(other->Mus);
        Ex = other->Ex;
        Ec = other->Ec;
        firstOrb = other->firstOrb;
        lastOrb = other->lastOrb;
        f_diag = other->f_diag;
        C=other->C;
        HJK=other->HJK;
        P0=other->P0;
        f_diagb=other->f_diagb;
        V0=other->V0;
        V=other->V;
        asp=other->asp;
        sH=other->sH;
        sHJK=other->sHJK;
        old_fock=other->old_fock;
        old_p=other->old_p;
        Rho=other->Rho;
        NewRho=other->NewRho;
        RhoM12=other->RhoM12;
        Mus->InitializeExpectation(Rho);
        if (Corr || params["BBGKY"] || params["RIFock"] || params["TDCIS"])
            *ee2 = *other->ee2;
        cout << "Sync'd Mean-Field Energy: " << MeanFieldEnergy(Rho,V,HJK) << endl;
    }
    
    void Save() const
    {
        Mus->save(logprefix);
        if (f_diag.n_elem)
            f_diag.save(logprefix+"f_diag");
        if (C.n_elem)
            C.save(logprefix+"C");
        if (HJK.n_elem)
            HJK.save(logprefix+"HJK");
        if (P0.n_elem)
            P0.save(logprefix+"P0");
        if (f_diagb.n_elem)
            f_diagb.save(logprefix+"f_diagb");
        if (V0.n_elem)
            V0.save(logprefix+"V0");
        if (V.n_elem)
            V.save(logprefix+"V");
        if (sH.n_elem)
            sH.save(logprefix+"sH");
        if (sHJK.n_elem)
            sHJK.save(logprefix+"sHJK");
        if (old_fock.n_elem)
            old_fock.save(logprefix+"old_fock");
        if (old_p.n_elem)
            old_p.save(logprefix+"old_p");
        if (Rho.n_elem)
            Rho.save(logprefix+"Rho");
        if (NewRho.n_elem)
            NewRho.save(logprefix+"NewRho");
        if (RhoM12.n_elem)
            RhoM12.save(logprefix+"RhoM12");
    }
    
	// P_ is in the current fock eigenbasis && will be updated.
    // V (the lowdin to fock transformation) is updated.
    // Fock is in the AO basis && will be updated.
    // The matrix returned is the Rotation matrix into the new fock eigenbasis from the old fock eigenbasis.
	cx_mat UpdateLiouvillian(vec& f_diag_, cx_mat& HJK, cx_mat& P_, bool Rebuild=true)
	{
        cx_mat Cprime(P_);
		if (Print>3)
			P_.print("P_ in update.");
		int n = the_scf->NOrb;
		int n_e = the_scf->NOcc;
        
		cx_mat Rhob = FullRho(P_); // Makes an X-basis density
        
		if (Print>3)
			Rhob.print("Rhob");
        
		if (Rebuild)
		{
			arma::cx_mat tmp;
            if (!ActiveSpace)
                Cprime = the_scf->UpdateLiouvillian(Ex, Ec, Scratch, f_diag_,V,HJK,tmp,tmp,Rhob,old_fock,old_p,IncFock && !(n_fock%100==0));
			else
                Cprime = the_scf->UpdateLiouvillian_ActiveSpace(Ex, Ec, Scratch,f_diag_,V,asp.Vs,HJK,Rhob,asp.P0cx,asp.P_proj,asp.Q_proj,asp.Ps_proj,asp.Qs_proj,old_fock,old_p,IncFock && !(n_fock%100==0));
            
            if (Stabilize)
                Posify(Rhob,the_scf->NOcc);
            
            if (!ActiveSpace)
            {
                P_ = V.t()*Rhob*V;
                asp.Vs=V;
            }
            else
            {
                // Rebuild overall V-matrix.
                P_ = asp.Vs.t()*Rhob*asp.Vs;
            }
            
            asp.Csub=the_scf->X*asp.Vs;
            
            // Update Dipole Matrices Into current fock basis.
            Mus->update(asp.Csub);
            // Update TCL.
            if (TCLOn && MyTCL != NULL)
                MyTCL->update(asp.Csub,f_diag_);
		}
        else if (Stabilize)
			Posify(Rhob,the_scf->NOcc);
        if (!is_finite(Rhob))
        {
            cout << "Posify returned garbage... " << endl;
            throw 1;
        }
        
        if (Print>3)
            P_.print("P_");
		
        return Cprime;
	}
    
	double VonNeumannEntropy(cx_mat& rho,bool print=false) const
	{
		vec eigval;
		cx_mat Cprime;
		if (!eig_sym(eigval, Cprime, rho, "dc"))
		{
			return 0.0;
		}
		if (print)
			eigval.st().print("Natural Occs");
		vec omp = 1.0-eigval;
		vec olnp = log(omp);
		vec lnp = log(eigval);
		for (int i=0; i<eigval.n_elem; ++i)
		{
			if (eigval(i)<pow(10.0,-12.0))
				lnp(i)=0.0;
			if (omp(i)<pow(10.0,-12.0))
				olnp(i)=0.0;
		}
		return -2.0*(dot(eigval,lnp)+dot(omp,olnp));
	}
	
	double MeanFieldEnergy(cx_mat& rho, cx_mat& V, cx_mat& HJK) const
	{
        cx_mat Plao;
        if (!ActiveSpace)
        {
            Plao = V*rho*V.t();
            arma::cx_mat Pao = the_scf->X.t()*Plao*the_scf->X;
            double Etot = real(dot(Pao,the_scf->H)+ dot(Pao,HJK))+ the_scf->nuclear_energy()+the_scf->Ec+the_scf->Ex;
            return Etot;
        }
        else
        {
            Plao = asp.P0cx+asp.Vs*rho*asp.Vs.t(); // asp.P0cx is in the x basis.
            arma::cx_mat Pao = the_scf->X.t()*Plao*the_scf->X;
            double Etot = real(dot(Pao,the_scf->H)+ dot(Pao,HJK))+ the_scf->nuclear_energy()+the_scf->Ec+the_scf->Ex;
            return Etot;
        }
	}
	
	// Ie: with Aufbau occupations.
	double HundEnergy(cx_mat& rho, cx_mat& V, cx_mat& HJK) const
	{
		arma::cx_mat PTemp(the_scf->NOrb,the_scf->NOrb);
		PTemp.eye();
		PTemp.submat(the_scf->NOcc,the_scf->NOcc,the_scf->NOrb-1,the_scf->NOrb-1) *= 0.0;
		arma::cx_mat Plao = V*PTemp*(V.t());
		arma::cx_mat Pao = the_scf->X.t()*Plao*the_scf->X;
		double Etot = real(dot(Pao,the_scf->H)+ dot(Pao,HJK))+ the_scf->nuclear_energy()+the_scf->Ec+the_scf->Ex;
		return Etot;
	}
	
    void WriteMoldenATOMS_Repositioned(FILE *fp) const
    {

        // Write the [Atoms] section containing the geometry.
        if (!fp) fp = stdout;
        
        int NAtoms = rem_read(REM_NATOMS);
        
        int *AtNo;
        double *Carts3,*xyz;
        get_carts(NULL,&Carts3,&AtNo,NULL);
        xyz = QAllocDouble(3*NAtoms);
        VRcopy(xyz,Carts3,3*NAtoms);
        
        bool useBohr = (rem_read(REM_INPUT_BOHR) == 1) ? true : false;
        if (!useBohr) VRscale(xyz,3*NAtoms,ConvFac(BOHRS_TO_ANGSTROMS));
        
        fprintf(fp,"[Molden Format]\n[Atoms] ");
        if (useBohr)
            fprintf(fp,"(AU)\n");
        else
            fprintf(fp,"(Angs)\n");
        
        double xm=0;
        double ym=0;
        double zm=0;
        for (int i = 0; i < NAtoms; i++){
            xm += xyz[3*i];
            ym += xyz[3*i+1];
            zm += xyz[3*i+2];
        }
        
        for (int i = 0; i < NAtoms; i++){
            String AtSymb = AtomicSymbol(AtNo[i]);
            char *atsymb = (char*)AtSymb;
            fprintf(fp,"%3s %6d %4d %15.8f %15.8f %15.8f\n",atsymb,
                    i+1,AtNo[i],xyz[3*i]-xm/NAtoms,xyz[3*i+1]-ym/NAtoms,xyz[3*i+2]-zm/NAtoms);
        }
        QFree(xyz);
    }
    
	// Allows for variable occupation numbers which are correctly plotted in GABEDIT.
	void WriteMoldenDiffDen(double *jCA, double *jCB, double *jEA, double *jEB, double *occa, double* occb,
							int NOccA,    int NVirA,    int NOccB,    int NVirB, FILE *fp, bool Rest=false) const
	{
		int NOccAPrt=NOccA;
		int NOccBPrt=NOccA;
		int NVirAPrt=NVirA;
		int NVirBPrt=NVirA;
		if (Rest)
			NOccBPrt = NVirBPrt = 0;
		//
		// Write the [MO] section using a "general" (i.e., user-specified)
		// set of MO coefficients.
		//
		// fp = destination file (opened by the caller).  Defaults
		//      to NULL for stdout.
		//
		// jCA,jCB = alpha/beta MO coefficients
		// jEA,jEB = alpha/beta orbital energies
		//
		// NOccA,NVirA,NOccB,NVirB
		//       = no. of occupied/virtual alpha/beta orbitals that are
		//         contained in the input jC.
		//
		// NOccAPrt = no. alpha occupieds to output (HOMO, HOMO-1, HOMO-2, ...)
		// NVirAPrt = no. alpha virtuals to output (LUMO, LUMO+1, LUMO+2, ...)
		// NOccBPrt, NVirBPrt = no. beta occupieds/virtuals to output
		//
		if (!fp) fp = stdout;
		int NBasis = bSetMgr.crntShlsStats(STAT_NBASIS);
		int NBas6D = bSetMgr.crntShlsStats(STAT_NBAS6D);
		int NAlpha = rem_read(REM_NALPHA);
		int NBeta  = rem_read(REM_NBETA);
		
		BasisSet basis(DEF_ID);
		LOGICAL *pureL = basis.deciPC();
		bool pureD = pureL[2],
		pureF = pureL[3],
		pureG = pureL[4];
		
		int NAtoms;
		if (rem_read(REM_QM_MM_INTERFACE) <= 0)
			NAtoms = rem_read(REM_NATOMS);
		else
			NAtoms = rem_read(REM_NATOMS_MODEL);
		
		int *AtNo,LMin,LMax,NShells,nfunc;
		int bCode = basis.code();
		get_carts(NULL,NULL,&AtNo,NULL);
		BasisID* BID = new BasisID [NAtoms];
		for (INTEGER i = 0; i < NAtoms; ++i)
			BID[i] = BasisID(AtNo[i],i+1,bCode);
		MolecularBasis molBasis(BID,NAtoms);
		delete [] BID;
		
		// Check that there are no basis functions with L > 3.
		// Molden supports L=4, but I don't think it's working yet here.
		
		int LMaxOverall=0;  // LMax for the entire basis
		for (int iAtom=0; iAtom < NAtoms; iAtom++){
			NShells = molBasis.pAtomicBasis(iAtom)->NumShells();
			for (int i=0; i < NShells; i++){
				LMax = molBasis.pAtomicBasis(iAtom)->LMax(i);
				if (LMax > 4){
					cout << "Molden output is not available for angular "
					<< "momentum beyond G\n";
					QWarn("Skipping Molden MO output");
					return;
				}
				else if (LMax > LMaxOverall)
					LMaxOverall = LMax;
			}
		}
		
		fprintf(fp,"[MO]\n");
		
		int *index = QAllocINTEGER(NBasis);
		getIndicesForMoldenMOs(index);
		
		if (NOccAPrt > 0){
			if (NOccAPrt > NOccA) NOccAPrt=NOccA; // that's all we have
			for (int i=NOccA-NOccAPrt; i < NOccA; i++){
				fprintf(fp,"Sym=X\nEne= %.6f\n",jEA[i]);
				fprintf(fp,"Spin=Alpha\nOccup=%.6f\n",occa[i]);
				for (int mu=0; mu < NBasis; mu++)
					fprintf(fp,"%6d%22.12e\n",mu+1,jCA[i*NBasis+index[mu]]);
			}
		}
		if (NVirAPrt > 0){
			if (NVirAPrt > NVirA) NVirAPrt=NVirA;
			for (int i=NOccA; i < NOccA+NVirAPrt; i++){
				fprintf(fp,"Sym=X\nEne= %.6f\n",jEA[i]);
				fprintf(fp,"Spin=Alpha\nOccup=%.6f\n",occa[i]);
				for (int mu=0; mu < NBasis; mu++)
					fprintf(fp,"%6d%22.12e\n",mu+1,jCA[i*NBasis+index[mu]]);
			}
		}
		if (NOccBPrt > 0){
			if (NOccBPrt > NOccB) NOccBPrt=NOccB;
			for (int i=NOccB-NOccBPrt; i < NOccB; i++){
				fprintf(fp,"Sym=X\nEne= %.6f\n",jEB[i]);
				fprintf(fp,"Spin=Beta \nOccup=%.6f\n",occb[i]);
				for (int mu=0; mu < NBasis; mu++)
					fprintf(fp,"%6d%22.12e\n",mu+1,jCB[i*NBasis+index[mu]]);
			}
		}
		if (NVirBPrt > 0){
			if (NVirBPrt > NVirB) NVirBPrt=NVirB;
			for (int i=NOccB; i < NOccB+NVirBPrt; i++){
				fprintf(fp,"Sym=X\nEne= %.6f\n",jEB[i]);
				fprintf(fp,"Spin=Beta \nOccup=%.6f\n",occb[i]);
				for (int mu=0; mu < NBasis; mu++)
					fprintf(fp,"%6d%22.12e\n",mu+1,jCB[i*NBasis+index[mu]]);
			}
		}
		
		// Molden assumes cartesian gaussians by default, so set an
		// appropriate flag in the case of pure (spherical) gaussians
		
		String purecart;
		bool needFlag=false;
		if (LMaxOverall == 2 && pureD){
			needFlag=true;
			purecart = "[5D]";
		}
		else if (LMaxOverall == 3){
			if (pureD || pureF) needFlag = true;
			
			if (pureD && pureF)
				purecart = "[5D7F]";
			else if (pureD && !pureF)
				purecart = "[5D10F]";
			else if (!pureD && pureF)
				purecart = "[7F]";
		}
		if (needFlag) fprintf(fp,"%s\n",(char*)purecart);
		QFree(index);
	}
	
    // the following routine is okay for gabedit (but only puts one density in the file.)
	void WriteEH(const cx_mat& rho,const mat& P0,const cx_mat& V, double t, int& Dest) const
	{
		int N = the_scf->NOrb;
		mat eigv(N,N);
		vec eige(N);
		// Generate the AO density.
		arma::cx_mat Plao = FullRho(rho);
		arma::mat Pao = real(the_scf->X.t()*Plao*the_scf->X);
        
		//Check trace:
		cout << "Writing Density Trace: " << trace(Pao*the_scf->AOS) ;
		Pao = Pao - P0;
		cout << " Difference Trace: " << trace(Pao*the_scf->AOS) << endl;
		// Now only the real part of this density can be responsible for n(r) generate the lowdin orbitals of that && write them to the file.
        
		eig_sym(eige, eigv, Pao*the_scf->AOS);
		eige.st().print("Difference Noccs:");

        for (int i=0;i<the_scf->NOrb; ++i)
            for (int j=0;j<the_scf->NOrb; ++j)
                if (!is_finite(eigv(i,j)))
                    eigv(i,j)=0.0;
		eigv = eigv.t(); // now has dimensions MOxAO
		int h = the_scf->NOcc-1; int l = h+1;
		int nes=0; int nhs=0;
		vec ho(N), eo(N), e2o(N);
		mat hc(N,N); mat ec(N,N);
		ho.zeros(); eo.zeros();
		hc.zeros(); ec.zeros();
		for (int i=0;i<the_scf->NOrb; ++i)
		{
            if (!is_finite(eige(i)))
                continue;
			if (eige(i)<0.0)
			{
				ho(i) = eige(i);
				hc.row(i) += eigv.row(i);
			}
			else
			{
				eo(i) = eige(i);
				ec.row(i) += eigv.row(i);
			}
		}
		char str[12];
		sprintf(str, "%i", (int)Dest);
		std::string name(str);
		cout << "|electron| " << sum(eo) << " |hole| " << sum(ho) << endl;
        {
            String fileName  = String("logs/")+String(name.c_str())+String("er.molden");
            FILE *fp = fopen(fileName.c_str(),"w"); //QOpen(fileName,"w");
            WriteMoldenATOMS_Repositioned(fp);
            WriteMoldenGTO(fp);
            ec = ec.t();
            vec dTs(N); dTs.zeros();
            dTs(0)=t;
            WriteMoldenDiffDen(ec.memptr(),ec.memptr(),dTs.memptr(),dTs.memptr(),eo.memptr(),eo.memptr(),
                               the_scf->NOcc,the_scf->NOrb-the_scf->NOcc,the_scf->NOcc,the_scf->NOrb-the_scf->NOcc,fp);
            fclose(fp);
        }
        {
            String fileName  = String("logs/")+String(name.c_str())+String("hr.molden");
            FILE *fp = fopen(fileName.c_str(),"w"); //QOpen(fileName,"w");
            WriteMoldenATOMS_Repositioned(fp);
            WriteMoldenGTO(fp);
            hc = hc.t();
            ho *= -1.0;
            vec dTs(N); dTs.zeros();
            dTs(0)=t;
            WriteMoldenDiffDen(hc.memptr(),hc.memptr(),dTs.memptr(),dTs.memptr(),ho.memptr(),ho.memptr(),
                               the_scf->NOcc,the_scf->NOrb-the_scf->NOcc,the_scf->NOcc,the_scf->NOrb-the_scf->NOcc,fp);
            fclose(fp);
        }
		Dest++;
	}
	   
	void Posify(cx_mat& rho,int N) const
	{
		int n = rho.n_rows;
		if (!is_finite(rho) || !rho.n_elem || rho.n_rows != rho.n_cols)
		{
            cout << " Posify passed garbage" << endl;
            rho.print("rho");
            throw 1;
		}
		vec eigval;
		cx_mat u;
		cx_mat Rhob(rho); Rhob.zeros();
		Rhob.eye();
		Rhob.submat(N,N,rho.n_cols-1,rho.n_cols-1) *= 0.0;
		//rho.diag().st().print("input rho");
		rho -= Rhob; // Difference density.
		rho = 0.5*(rho+rho.t());
        
		if (abs(trace(rho)) < pow(10.0,-13.0))
		{
			rho+=Rhob;
			if (Print)
				cout << " too small, not stabilizing trace " << endl;
		}
		else
		{
			// Trace && e-h symmetry correction.
			if (!eig_sym(eigval, u, rho, "std"))
			{
				rho.print("rho-eig sym failed.");
                throw 1;
			}
			double ne = 0.0; double nh=0.0;
			vec IsHole(rho.n_cols); IsHole.zeros();
			vec IsE(rho.n_cols); IsE.zeros();
			for (int i=0; i<rho.n_cols;++i)
			{
				if (abs(eigval(i))>1.0)
				{
					eigval(i) = ((0 < eigval(i)) - (eigval(i) < 0))*1.0;
				}
				if (eigval(i)<0) // it's a hole
				{
					IsHole(i) = 1.0;
					nh+=eigval(i);
				}
				else
				{
					IsE(i) = 1.0;
					ne+=eigval(i);
				}
			}
			if (abs(sum(eigval%IsHole)) < abs(sum(eigval%IsE)))
			{
				// fewer holes than electrons, scale the electrons down so that they match
                if ( abs(sum(eigval%IsE)) != 0.0)
                {
                    IsE *= abs(sum(eigval%IsHole))/abs(sum(eigval%IsE));
                    eigval = (eigval%(IsE+IsHole));
                }
			}
			else
			{
				// Fewer electrons than holes, scale the holes down.
                if ( abs(sum(sum(eigval%IsHole))) != 0.0)
                {
                    IsHole *= abs(sum(eigval%IsE))/abs(sum(eigval%IsHole));
                    eigval = (eigval%(IsE+IsHole));
                }
			}
			if (Print)
				cout << " Difference Corrected to: " << sum(eigval);
			rho=u*diagmat(eigval)*u.t()+Rhob;
			if (Print)
				cout << " Trace Corrected to: " << trace(rho) << endl;
		}
		// positivity correction.
		if (Stabilize > 1)
		{
			vec eigval;
			cx_mat u;
			if (!eig_sym(eigval, u, rho, "std"))
			{
				cout << "eig_sym failed" << endl;
				rho.print("rho");
				throw 1;
			}
			if (Print)
				eigval.st().print("Nat Occs. before"); // In ascending order.
			for (int i=0; i<rho.n_cols;++i)
			{
				if (eigval(i)<0)
					eigval(i)*=0.0;
				else if (eigval(i)>1.0)
					eigval(i)=1.0;
			}
            double missing = (double)N-sum(eigval);
            if (abs(missing) < pow(10.0,-10.0))
                return;
			if (n-N-1<0)
			{
				rho=u*diagmat(eigval)*u.t();
				return;
			}
			if (Print)
				cout << "Positivity Error:" << missing;
			if (missing < 0.0 && eigval(n-N-1)+missing > 0.0)
				eigval(n-N-1) += missing; // Subtract from lumo-like
			else if (missing < 0.0)
				eigval(n-N) += missing;  // Subtract from homo-like
			else if (missing > 0.0 && eigval(n-N)+missing < 1.0)
				eigval(n-N) += missing; //  add to homo-like
			else
				eigval(n-N-1) += missing; //  add to lumo-like
			if (Print)
				eigval.st().print("Nat Occs. after"); // In ascending order.
			missing = (double)N-sum(eigval);
            //			if (Print)
            //				cout << " corrected to: " << missing << endl;
			rho=u*diagmat(eigval)*u.t();
            // Finally recheck the diagonal is positive.
            for (int i=0; i<n;++i)
            {
                if (real(rho(i,i))<0.0)
                    rho(i,i)=0.0;
            }
            
		}
		return;
	}
    
    // Do a shifted/scaled fourier transform of polarization in each direction.
    mat Fourier(const mat& pol, double zoom = 1.0/20.0, int axis=0, bool dering=false) const
    {
        mat tore;
        double dt = pol(3,0)-pol(2,0);
        {
            int ndata = (pol.n_rows%2==0)? pol.n_rows : pol.n_rows-1; // Keep an even amount of data
            for (int r=1; r<ndata; ++r)
                if (pol(r,0)==0.0)
                    ndata =	(r%2==0)?r:r-1;  // Ignore trailing zeros.
            vec data = pol(arma::span(0,ndata-1),1+axis);
            data = data - accu(data)/ndata; // subtract any zero frequency component.
            
            if (dering)
            {
                double gam = log(0.00001/(ndata*ndata));
                vec de = linspace(0.0,ndata,ndata);
                de = exp(-1.0*de%de*gam);
                data = data%de;
            }
            
            cx_vec f(ndata); f.zeros();
            //cx_vec ftmp(ndata); ftmp.ones(); // Holds f^n
            vec fr(ndata/2), fi(ndata/2); fr.zeros(); fi.zeros();
            
            for (int r=0; r<ndata; ++r)
                f(r) = exp(std::complex<double>(0.0,2.0*M_PI)*zoom*(r-1.0)/ndata);
            cx_vec ftmp(ndata); ftmp.ones();
            for (int i=0; i<ndata/2; ++i) // I only even generate the positive frequencies.
            {
                ftmp = f%ftmp;
                fr(i) = real(sum(ftmp%data));
                fi(i) = imag(sum(ftmp%data));
            }
            tore.resize(ndata/2,3);
            for (int i=0; i<ndata/2; ++i)
                tore(i,0) = M_PI*(27.2113*DipolesEvery/dt)*zoom*i/ndata;
            tore.col(1) = -1.0*fi;
            tore.col(2) = fr;
        }
        return tore;
    }
    
    TDSCF(rks* the_scf_,const std::map<std::string,double> Params_ = std::map<std::string,double>(), TDSCF_Temporaries* Scratch_ = NULL) : the_scf(the_scf_), logprefix("./logs/"), n_ao(the_scf_->N),n_mo(the_scf_->NOrb), n_occ(the_scf_->NOcc), n_e(the_scf_->NOcc), MyTCL(NULL), npol_col(10), ee2(NULL)
	{
        if (Scratch_!=NULL)
            Scratch = Scratch_;
        else
            Scratch = new TDSCF_Temporaries();
        
        if (rem_read(REM_TDCI))
            return;
        
		cout << endl;
		cout << "===================================" << endl;
		cout << "|  Realtime TDSCF module          |" << endl;
		cout << "===================================" << endl;
		cout << "| J. Parkhill, T. Nguyen          |" << endl;
		cout << "| J. Koh, J. Herr,  K. Yao        |" << endl;
		cout << "===================================" << endl;
        cout << "| Refs: 10.1021/acs.jctc.5b00262  |" << endl;
        cout << "|       10.1063/1.4916822         |" << endl;
		cout << "===================================" << endl;
		n = n_mo; cout << "n_ao:" << n_ao << " n_mo " << n_mo << " n_e/2 " << n_occ+1 << endl;
		n_fock = 0; // number of fock updates.
        
		params = ReadParameters(Params_);
        if (!RunDynamics)
            return;
        
		V.resize(n,n); V.eye(); // The Change of basis between the Lowdin AO's && the current fock eigenbasis (in which everything is propagated)
		asp.Vs.resize(n,n); asp.Vs.eye(); // The Change of basis between the Lowdin AO's && the current fock eigenbasis (in which everything is propagated)
        
        C.resize(n,n); C.zeros(); // = X*V.
        HJK.resize(n,n); HJK.zeros(); // The Change of basis between the Lowdin AO's && the current fock eigenbasis (in which everything is propagated)
        
        P0=the_scf->P; // the initial density matrix (ao basis)
        f_diagb.resize(n_mo);
		cx_mat mu_xb(n_ao,n_ao), mu_yb(n_ao,n_ao), mu_zb(n_ao,n_ao); // Stored in the ao representation && transformed on demand.
		{
			cx_mat Rhob(n,n); Rhob.zeros();
			InitializeLiouvillian(f_diagb,V,mu_xb,mu_yb,mu_zb);
            f_diag = f_diagb;
			C = the_scf->X*V;
		}
		V0=V; // the initial mo-coeffs in terms of Lowdin (x-basis).
		
        // Note the mu_x,y etc. are provided to FieldMatrices in the X basis.
        if (ApplyNoOscField)
            Mus = new StaticField(mu_xb, mu_yb, mu_zb, FieldVector, FieldAmplitude, Tau, tOn);
        else
            Mus = new OpticalField(mu_xb, mu_yb, mu_zb, FieldVector, ApplyImpulse, ApplyCw, FieldAmplitude, FieldFreq, Tau, tOn, ApplyImpulse2, FieldAmplitude2, FieldFreq2, Tau2, tOn2);
        
        // Plot the Basis' approximation to the x field if you so desire it.
#ifndef RELEASE
        PlotFields();
#endif
        
		// figure out which block of orbitals we'll propagate within.
		firstOrb = 0;
		lastOrb = n-1;
        f_diag.resize(n); f_diag.zeros();
        asp.Csub = the_scf->X*V;
        
		if (ActiveSpace)
            SetupActiveSpace();
		
		cout << "Propagating space contains: " << n_e*2 << " singlet fermions in " << n << " orbitals "<< endl;
        
        Rho.resize(n,n); NewRho.resize(n,n);
        Rho.zeros(); NewRho.zeros();
		Rho.eye(); Rho.submat(n_e,n_e,n-1,n-1) *= 0.0;
		
        if (ActiveSpace)
        {
            UpdateLiouvillian( f_diag, HJK, Rho);
            sH = asp.Vs.t()*(the_scf->X*(the_scf->H)*the_scf->X.t())*asp.Vs;
            sHJK = asp.Vs.t()*(the_scf->X*HJK*the_scf->X.t())*asp.Vs;
            cx_mat Pt = the_scf->X.t()*asp.P0cx*the_scf->X;
            asp.ECore = real(dot(Pt,the_scf->H)+ dot(Pt,HJK));
            cout << std::setprecision(9) << "Mean field Energy of initial state (Active Space): " << MeanFieldEnergy(Rho,V,HJK) << endl;
        }
        else
        {
            f_diag = f_diagb;
            cout << std::setprecision(9) << "Mean field Energy of initial state: " << MeanFieldEnergy(Rho,V,HJK) << endl;
        }
        cout << "Koopman's Excitation estimate: " << f_diag(n_e)-f_diag(n_e-1) << " (eH) " <<  27.2113*(f_diag(n_e)-f_diag(n_e-1)) << " (eV) "<< endl;
        
        if (TCLOn && MyTCL == NULL)
            MyTCL = new TCLMatrices(n,n_ao,asp.Csub,f_diag,n_e,params);
        
        if (StartFromSetman>0)
            SetmanDensity();
        else if (InitialCondition == 1)
        {
            cout << "WARNING -- WARNING" << endl;
            cout << "Initalizing into a particle hole excitation out of: " << n_e-1 << " && rebuilding the fock matrix (once)" << endl;
            Rho(n_e-1,n_e-1)=0.0;    Rho(n_e,n_e)=1.0; // initialize into a particle hole excitation
        }
        
        if (UpdateFockMatrix)
        {
            UpdateLiouvillian(f_diag, HJK, Rho); // update the eigenvalues && TCL.
        }
        
		C = the_scf->X*V;
		asp.Csub=C.submat(0,firstOrb,n_ao-1,lastOrb);
		cout << "Finished Calculating the Initial Liouvillian, Energy: " << MeanFieldEnergy(Rho,V,HJK) << endl;
		f_diag.st().print("Initial Eigenvalues:");
		cout << "Excitation Gap: " << (f_diag(n_e)-f_diag(n_e-1)) << " Eh : " << (f_diag(n_e)-f_diag(n_e-1))*27.2113 << "eV" << endl;
		cout << "Condition of Fock Matrix: " << 1.0/(f_diag(n-1)-f_diag(0)) << " timestep (should be smaller) " << dt << endl;
		Rho.diag().st().print("Populations after first Fock build:");
        cout << "Idempotency: " << accu(Rho-Rho*Rho) <<  endl;
		Mus->update(asp.Csub);
		cout << "Homo->Lumo transition strength: " << Mus->muxo(n_e-1,n_e) << endl;
        
		Mus->InitializeExpectation(Rho);
        t=0; loop=0; loop2=0; loop3=0;
        RhoM12=Rho;
        
        if (!Restart)
        {
            Pol.resize(5000,npol_col);
            Pol.zeros(); // Polarization Energy, Trace, Gap, Entropy, HOMO-LUMO Coherence.
        }
        else
            ReadRestart();
        
		if (SavePopulations)
		{
			Pops = mat(100,n+1+2);//KEVINEVIN +2
			Pops.zeros();
		}
		if (SaveFockEnergies)
		{
			FockEs = mat(100,n+1);
			FockEs.zeros();
		}
        
        if (Corr || params["TDCIS"]  || params["BBGKY"] || params["RIFock"])
        {
            params["n_mo"] = n_mo;
            params["n_occ"] = n_occ;
#if HASEE2
            threading_policy::enable_omp_only();
            if (ActiveSpace)
            {
                if (TCLOn)
                    ee2 = new EE2(n_mo,n_occ, the_scf, Rho, V, params,&asp,MyTCL);
                else
                    ee2 = new EE2(n_mo,n_occ, the_scf, Rho, V, params,&asp);
            }
            else
            {
                if (TCLOn)
                    ee2 = new EE2(n_mo,n_occ, the_scf, Rho, V, params, NULL, MyTCL);
                else
                    ee2 = new EE2(n_mo,n_occ, the_scf, Rho, V, params, NULL, NULL, Mus);
            }
#endif
            if (!params["ActiveSpace"])
            {
                Posify(Rho, n_e);
                RhoM12=Rho;
                C=the_scf->X*V;
                Mus->update(C);
                Mus->InitializeExpectation(Rho);
            }
            else
            {
                Posify(Rho, asp.ne);
                RhoM12=Rho;
                Rho.print("Initial Rho");
                Mus->update(asp.Csub);
                Mus->InitializeExpectation(Rho);
            }
		}
        
        LastFockPopulations.resize(n);
        LastFockEnergies.resize(n);
        LastFockPopulations = Rho.diag();
        LastFockEnergies = f_diag;
        Entropy = VonNeumannEntropy(Rho,true);
        if ((UpdateMarkovRates ||  StartFromSetman) && TCLOn && MyTCL != NULL)
        {
            MyTCL->rateUpdate();
            cout << "Gap in units of KbT: " << (f_diag(n_e)-f_diag(n_e-1))*MyTCL->Beta  << endl;
        }
        
		cout << "=============================================" << endl;
        if (SerialNumber)
            cout << "Initalization of TDSCF SerialNumber " << SerialNumber << " complete" << endl;
        else
            cout << "Initalization of TDSCF complete" << endl;
		cout << "=============================================" << endl;
		wallt0 = time(0); // This is in wall-seconds.
        if (!rem_read(REM_RTPUMPPROBE))
            Propagate();
		return;
	}
    
    bool Propagate()
    {
        
        if (MaxIter)
            if (loop>MaxIter)
            {
                cout << "Exiting Because MaxIter exceeded. SerialNumber"<< SerialNumber << " Iter " << loop << " of " << MaxIter << endl;
                return false;
            }
        
        bool IsOn;
        double WallHrPerPs=0;
        if (Corr && !params["RIFock"])
            cout << "---Corr TDHF---" << endl;
        else if (UpdateFockMatrix)
            cout << "---MMU TDHF---" << endl;
        else
            cout << "---Frozen Orbitals---" << endl;

        Rho.diag().st().print("Initial Populations:");
        
        
        while(true)
        {
            // The Timestep ---------------------------------------------------------------
            if (Corr || params["TDCIS"]  || params["BBGKY"] || params["RIFock"])
            {
                if (params["Print"])
                    cout << "CorrTimestep." << endl;
#if HASEE2
                threading_policy::enable_omp_only();
                Energy = ee2->step(the_scf, Mus, f_diag, V, Rho, RhoM12, t, dt, IsOn);
                threading_policy::pop();
#else
                cout << "No ee2." << endl; throw;
#endif
                if (Stabilize)
                {
                    if (params["ActiveSpace"])
                    {
                        Posify(Rho, asp.ne);
                        Posify(RhoM12, asp.ne);
                    }
                    else
                    {
                        Posify(Rho, n_e);
                        Posify(RhoM12, n_e);
                    }
                }
                n_fock++;
                // Is a rate Rebuild Called for?
                vec tmp = f_diag - LastFockEnergies;
                double NewRates = norm(tmp);
                if (NewRates > RateBuildThresh && UpdateMarkovRates && MyTCL != NULL)
                {
                    MyTCL->rateUpdate();
                    LastFockEnergies = f_diag;
                }
                if (params["Print"])
                    cout << "~CorrTimestep." << endl;
            }
            else if (UpdateFockMatrix)
            {
                MMUTStep(f_diag, HJK, V, Rho, RhoM12, t, dt,IsOn);
                n_fock++;
                // Is a rate Rebuild Called for?
                //f_diag.st().print(); 
                vec tmp = f_diag - LastFockEnergies;
                double NewRates = norm(tmp);
                if (NewRates > RateBuildThresh && UpdateMarkovRates && MyTCL != NULL)
                {
                    MyTCL->rateUpdate();
                    LastFockEnergies = f_diag;
                }
            }
            else
            {
                Split_RK4_Step(f_diag, Rho, NewRho, t, dt,IsOn);
                Rho=0.5*(NewRho+NewRho.t());
                RhoM12 = Rho;
            }
            LastFockPopulations = Rho.diag();
            t+=dt;
            // Logging ---------------------------------------------------------------
            if (SaveFockEvery)
            {
                if (n_fock%SaveFockEvery == 0 && UpdateFockMatrix)
                {
                    int rw=n_fock/SaveFockEvery;
                    if (SaveFockEnergies)
                    {
                        FockEs(rw,0) = t;
                        if (FockEs.n_cols != f_diag.n_elem+1)
                            FockEs.resize(FockEs.n_rows,f_diag.n_elem+1);
                        FockEs.row(rw).cols(1,n) = real(f_diag.st());
                        if ( FockEs.n_rows-rw < 10)
                            FockEs.resize(FockEs.n_rows+2000,n+1);
                    }
                }
            }
            if (SavePopulations && (loop2%((int)params["SavePopulationsEvery"])==0))
            {
                if ((loop2%(int)params["SavePopulationsEvery"])==0)
                {
                    int rw;
                    for (rw=1;rw<Pops.n_rows; ++rw)
                        if (Pops(rw,0)==0)
                            break;
                    Pops(rw,0)=t;
                    
                    if (params["PrntPopulationType"] == 0.0)
                    {
                        cx_mat tmp;
                        if (ActiveSpace)
                            tmp=(asp.Vs*Rho*asp.Vs.t()) + asp.P0cx;
                        else
                            tmp=(V*Rho*V.t());
                        cx_mat RhoIn0 = V0.t()*tmp*V0; // Orig. Basis density matrix.
                        if (Pops.n_cols != RhoIn0.n_cols+1)
                            Pops.resize(Pops.n_rows,RhoIn0.n_cols+1+2);
                        Pops.row(rw).cols(1,RhoIn0.n_cols) = real(RhoIn0.diag().t());
                        Pops(rw,2+1) = real(RhoIn0(1,0));
                        Pops(rw,2+2) = imag(RhoIn0(1,0));
                        if ( Pops.n_rows-rw <10)
                            Pops.resize(Pops.n_rows+2000,5);// fixit later
                        
                    }
                    else if (params["PrntPopulationType"] == 1.0)
                    {
                        vec nocs; cx_mat nos;
                        eig_sym(nocs, nos, Rho);
                        if (Pops.n_cols != nocs.n_elem+1)
                            Pops.resize(Pops.n_rows+2000,nocs.n_elem+1);
                        Pops.row(rw).cols(1,nocs.n_elem) = nocs.st();
                        if ( Pops.n_rows-rw <10)
                            Pops.resize(Pops.n_rows+2000,nocs.n_elem+1);
                    }
                    
                }
            }
            if (loop%DipolesEvery==0 && DipolesEvery)
            {
                Entropy = VonNeumannEntropy(Rho,Print>0);
                if (!(Corr || params["TDCIS"]  || params["BBGKY"] || params["RIFock"]) && n_ao*DipolesEvery)
                    Energy = MeanFieldEnergy(Rho,V,HJK);
                vec tmp = Mus->Expectation(Rho);
                Pol(loop2,0) = t; // in atomic units.
                Pol(loop2,1) = tmp(0);
                Pol(loop2,2) = tmp(1);
                Pol(loop2,3) = tmp(2);
                Pol(loop2,4) = Energy;
                Pol(loop2,5) = real(trace(Rho));
                Pol(loop2,6) = (f_diag(n_e)-f_diag(n_e-1));
                Pol(loop2,7) = Entropy;
                Pol(loop2,8) = real(Rho(n_e-1,n_e));
                Pol(loop2,9) = real(Rho(n_e-1,n_e-1));
#if HASEE2
                if (Corr)
                    Pol(loop2,9) = ee2->Ehf;
#endif
                if (Pol.n_rows-loop2<100)
                    Pol.resize(Pol.n_rows+20000,Pol.n_cols);
                loop2 +=1;
            }
            if (SaveEvery)
                if (loop%SaveEvery==0)
                {
                    real(Rho.diag().st()).print("Pops:");
                    
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
            if(FourierEvery)
                if (loop%FourierEvery==0 && loop > 10)
                {
                    mat four_x=Fourier(Pol,FourierZoom,0);
                    four_x.save(logprefix+"Fourier_x"+".csv", raw_ascii);
                    mat four_y=Fourier(Pol,FourierZoom,1);
                    four_y.save(logprefix+"Fourier_y"+".csv", raw_ascii);
                    mat four_z=Fourier(Pol,FourierZoom,2);
                    four_z.save(logprefix+"Fourier_z"+".csv", raw_ascii);
                }
            
            // Finish Iteration ---------------------------------------------------------------
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
                    if (SerialNumber)
                        cout << "P: " << SerialNumber << " ITER: " << loop << " T: " << std::setprecision(2) << t*FsPerAu << "(fs) dt " << dt*FsPerAu  << "(fs) Hr/Ps: " << WallHrPerPs << " - Lpsd/Rem.: " << Elapsed << ", " << Remaining << " (min) Tr.Dev: " << abs(n_e-trace(Rho)) << " Hrm: " << abs(accu(Rho-Rho.t())) << std::setprecision(6) << " Enrgy: " << Energy << " Entr: " << Entropy <<  " Fld " << IsOn << " NFk: " << n_fock << endl;
                    else
                        cout << "ITER: " << loop << " T: " << std::setprecision(2) << t*FsPerAu << "(fs)  dt " << dt*FsPerAu  << "(fs) Hr/Ps: " << WallHrPerPs << " - Lpsd/Rem.: " << Elapsed << ", " << Remaining << " (min) Tr.Dev: " << abs(n_e-trace(Rho)) << " Hrm: " << abs(accu(Rho-Rho.t()))  << std::setprecision(6) << " Enrgy: " << Energy << " Entr: " << Entropy  << " Fld " << IsOn << " NFk: " << n_fock << endl;
                    if (Print)
                        Rho.diag().st().print("Populations");
                    Mus->Expectation(Rho).st().print("Mu");
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
    
    ~TDSCF()
    {
        cout << "~TDSCF()" << endl;
        if (!params["CIS"])
        {
            UpdateLiouvillian(f_diag,HJK,Rho);
            f_diag.st().print("Final Eigenvalues");
            Rho.diag().st().print("Final Populations");
            cout << "FINAL MEAN FIELD ENERGY: " << Energy << endl;
            cout << "FINAL ENTROPY:" << Entropy << endl;
            if (MyTCL!=NULL)
            {
                cout << "FINAL ENTROPY (hartrees):" << Entropy/(MyTCL->Beta) << endl;
                cout << "FINAL THERMODYNAMIC ENERGY:" << Energy-Entropy/(MyTCL->Beta) << endl;
                delete MyTCL;
                MyTCL=NULL;
            }
            cout << "FINAL HUNDS MEAN FIELD ENERGY: " << HundEnergy(Rho,V,HJK) << endl;
            // Save P-Matrix in Lowdin basis.
            FullRho(Rho).save(logprefix+"Px_TDSCF",arma_binary);
        }
        if (Mus!=NULL)
        {
            delete Mus;
            Mus=NULL;
        }
#if HASEE2
		if (ee2 != NULL)
        {
			delete ee2;
            ee2=NULL;
        }
#endif
    }
    
    
    // This is for pump-probe.
    // Creates a special output files that contain:
    // Pumped dipole, Nonstationary dipole, Corrected dipole
    // && Fouriers of the above.
    void TransientAbsorptionCorrection(const TDSCF* NonSt, const TDSCF* GSA) const
    {
        cout << "Calculating Corrected TA Spectrum. It will go into path " << logprefix << endl;
        if (Pol.n_rows < loop2 ||  NonSt->Pol.n_rows < loop2)
        {
            cout << "Storage error " << loop2 << " : " << Pol.n_rows << " : " << NonSt->Pol.n_rows  << endl;
            throw;
        }
        
        // Store all three pol files for reference.
        Pol.save(logprefix+"Pol.csv",raw_ascii);
        NonSt->Pol.save(logprefix+"RefCorr.csv",raw_ascii);
        GSA->Pol.save(logprefix+"GS_Pol.csv",raw_ascii);
        
        mat PPol(loop2,4); // time , x1 y1 z1 ...
        PPol.zeros();
        PPol(arma::span(0,loop2-1), 0) = Pol(arma::span(0,loop2-1),0);
        PPol(arma::span(0,loop2-1), arma::span(1,3)) = Pol(arma::span(0,loop2-1), arma::span(1,3)) - NonSt->Pol(arma::span(0,loop2-1), arma::span(1,3)) - GSA->Pol(arma::span(0,loop2-1), arma::span(1,3));
        PPol.save(logprefix+"TAPol.csv",raw_ascii);
        
        mat gsa_x=Fourier(GSA->Pol,FourierZoom,0);
        gsa_x.save(logprefix+"Fourier_GS_x"+".csv", raw_ascii);
        mat gsa_y=Fourier(GSA->Pol,FourierZoom,1);
        gsa_y.save(logprefix+"Fourier_GS_y"+".csv", raw_ascii);
        mat gsa_z=Fourier(GSA->Pol,FourierZoom,2);
        gsa_y.save(logprefix+"Fourier_GS_z"+".csv", raw_ascii);
        
        mat four_x=Fourier(PPol,FourierZoom,0);
        four_x.save(logprefix+"Fourier_TA_x"+".csv", raw_ascii);
        mat four_y=Fourier(PPol,FourierZoom,1);
        four_y.save(logprefix+"Fourier_TA_y"+".csv", raw_ascii);
        mat four_z=Fourier(PPol,FourierZoom,2);
        four_z.save(logprefix+"Fourier_TA_z"+".csv", raw_ascii);
    }
    
};


//  Pump-probe job requires 5 propagations.
//  begin from a linear response excited state.
//  Propagate that state without Fock build for t_delay fs.
//  Allow the system to warm up (decohere) for a bit TDecoh
//  Begin three 10fs propagations:
//  1 pump & probe, 2 - pump no probe (to correct nonstationarity) 3 - ground state (to get Delta A.)
//  Finally the dipole is made from 1-(2+3)
inline void PumpProbeJob(rks* pr)
{
    std::map<std::string,double> NoField, xField, Rlx, Gs, Warm;
    TDSCF_Temporaries* Scratch = new TDSCF_Temporaries();

    // First the relaxation job.
    Rlx["ApplyImpulse"] = 0;
    Rlx["StartFromSetman"] = int(rem_read(REM_RTPPSTATE));
    Rlx["UpdateFockMatrix"] = 0;
    Rlx["SerialNumber"] = 1;
    Rlx["Stabilize"] = 2;
    Rlx["Decoherence"] = 1; // Will be shut off when the warming is complete.
    Rlx["SaveDipoles"] = 0;
    Rlx["DipolesEvery"] = 1000;
    Rlx["StatusEvery"] = 2000;
    Rlx["SaveEvery"] = 5000;
    Rlx["SavePopulations"] = 1;
    Rlx["SavePopulationsEvery"] = 1000;
    Rlx["FourierEvery"] = 0;
    Rlx["WriteDensities"] = 0;
    Rlx["SaveFockEvery"] = 0;
    Rlx["TCLOn"] = 1;
    Rlx["MaxIter"] = 0; // Perform an infinite number of propagations.
    
    NoField["ApplyImpulse"] = 0;
    NoField["Stabilize"] = 0;
    NoField["TCLOn"] = 0;
    NoField["StartFromSetman"] = 0;
    NoField["Stutter"] = 4500; // Pauses propagation every n iterations && does corrected fourier transform.
    NoField["SaveDipoles"] = 1;
    NoField["StatusEvery"] = 100;
    NoField["SaveEvery"] = 1000;
    NoField["WriteDensities"] = 0;
    NoField["FourierEvery"] = 0;
    NoField["SaveFockEnergies"] = 0;
    NoField["SaveFockEvery"] = 0;
    NoField["SerialNumber"] = 2;

    xField["Stutter"] = 4500;
    xField["ApplyImpulse"] = 1;
    xField["Stabilize"] = 0; 
    xField["TCLOn"] = 0;
    xField["StatusEvery"] = 100;
    xField["SaveEvery"] = 1000;
    xField["StartFromSetman"] = 0; // The final density of Rlx will be fed in.

    
    Gs["ApplyImpulse"] = 1;
    Gs["Stabilize"] = 0;
    Gs["TCLOn"] = 0;
    Gs["StartFromSetman"] = 0;
    Gs["Stutter"] = 4500;
    Gs["TCLOn"] = 0;
    Gs["SaveDipoles"] = 1;
    Gs["StatusEvery"] = 100;
    Gs["SaveEvery"] = 1000;
    Gs["WriteDensities"] = 0;
    Gs["FourierEvery"] = 0;
    Gs["SaveFockEnergies"] = 0;
    Gs["SaveFockEvery"] = 0;
    Gs["SerialNumber"] = 4;
    
    TDSCF* Gs_p = new TDSCF(pr,Gs,Scratch);
    double TSample = Gs_p->params["TSample"]/FsPerAu;
    double TDecoh = Gs_p->params["TDecoh"]/FsPerAu;
    
    // perform multiple increasing delays for indefinite time, && sample TA spectrum every 500fs for the MaxIter in TDSCF.prm
    if (TDecoh < 0.0)
    {
        Rlx["Decoherence"] = 1; // Will be shut off when the warming is complete.
        Rlx["dt"] = 0.45; // Enormous steps can be taken because only the populations are being updated.
        Rlx["Stutter"] = (int)(TSample/Rlx["dt"]);
    }
    else if (TDecoh > TSample)
    {
        cout << "Decoherence time must be less than sample time." << endl;
        TSample = TDecoh + 150.0/FsPerAu;
    }
    else
        Rlx["Stutter"] = (int)(TDecoh/Gs_p->params["dt"]);
    cout << "----------------------------------------------------" << endl;
    cout << "Beginning pump probe propagations... " << endl;
    cout << "TDecoh (au)" << TDecoh << " NSample: " << Rlx["Stutter"] << endl;
    cout << "TSample (au)" << TSample << endl;
    cout << "Excited Fraction " << Gs_p->params["ExcitedFraction"] << endl;
    cout << "----------------------------------------------------" << endl;
    
    int samples = 1;
    TDSCF* Rlx_p = new TDSCF(pr,Rlx,Scratch);
    // The zero delay part doesn't require any Rlx, || warming.
    {
        TDSCF* NonStationary_p = new TDSCF(pr,NoField,Scratch);
        xField["SerialNumber"] = 3; // The zero delay propagation is named 3.
        TDSCF* ToCorrect_p = new TDSCF(pr,xField,Scratch);
        NonStationary_p->Sync(Rlx_p);
        ToCorrect_p->Sync(Rlx_p);
        while (true)
        {
            if (!NonStationary_p->Propagate())
                break;
            Gs_p->Propagate();
            ToCorrect_p->Propagate();
            ToCorrect_p->TransientAbsorptionCorrection(NonStationary_p,Gs_p);
        }
        
        Gs_p->Save();

        delete NonStationary_p;
        delete ToCorrect_p;
    }
    
    while (true)
    {
        if (Rlx_p->t < TDecoh)
        {
            Rlx_p->MyTCL->Decoherence = true;
            cout << "----------------------------------------------------" << endl;
            cout << "Beginning Decoh ..." << endl;
            cout << "----------------------------------------------------" << endl;
            Rlx_p->Propagate();
            cout << "----------------------------------------------------" << endl;
            cout << "Decoh Over. Final Rho: " << endl;
            Rlx_p->Rho.print("Decohered Rho");
            cout << "----------------------------------------------------" << endl;
            Rlx["dt"] = 0.45; // Enormous steps can be taken because only the populations are being updated.
            Rlx_p->dt = Rlx["dt"];
            Rlx_p->MyTCL->Decoherence = false;
            Rlx["Stutter"] = (int)(TSample/Rlx["dt"]);
            Rlx_p->Stutter = (int)(TSample/Rlx["dt"]);
        }
        
        cout << "----------------------------------------------------" << endl;
        cout << "Beginning Finite Delay... tau = " << Rlx_p->t*FsPerAu << "(fs)" << endl;
        cout << "----------------------------------------------------" << endl;
        if (!Rlx_p->Propagate())
            break;
        
        Rlx_p->Save();
        
        cout << "----------------------------------------------------" << endl;
        cout << "Relaxation Complete for " << Rlx_p->t*FsPerAu << "(fs)" << endl;
        Rlx_p->Rho.print("Rho");
        cout << "----------------------------------------------------" << endl;
        // Grab the density from the warm job & pass it to these propagations.
        TDSCF* NonStationary_p = new TDSCF(pr,NoField,Scratch);
        xField["SerialNumber"] = (int)(Rlx_p->t*FsPerAu);
        TDSCF* ToCorrect_p = new TDSCF(pr,xField,Scratch);
        NonStationary_p->Sync(Rlx_p);
        ToCorrect_p->Sync(Rlx_p);
        while (true)
        {
            if (!NonStationary_p->Propagate())
                break;
            ToCorrect_p->Propagate();
            ToCorrect_p->TransientAbsorptionCorrection(NonStationary_p,Gs_p);
            NonStationary_p->Save();
            ToCorrect_p->Save();
        }
        samples++;
        delete NonStationary_p;
        delete ToCorrect_p;
    }
    
    delete Gs_p;
    delete Rlx_p;
    
    delete Scratch;
};

// Runs three propagations in each direction to create an isotropic Abs. Spec.
inline void IsotropicJob(rks* pr)
{
    std::map<std::string,double> x,y,z;
    TDSCF_Temporaries* Scratch = new TDSCF_Temporaries();
    
    x["ExDir"] = 1.0;
    y["ExDir"] = 0.0;
    z["ExDir"] = 0.0;
    x["EyDir"] = 0.0;
    y["EyDir"] = 1.0;
    z["EyDir"] = 0.0;
    x["EzDir"] = 0.0;
    y["EzDir"] = 0.0;
    z["EzDir"] = 1.0;
    x["Stutter"] = 4500;
    y["Stutter"] = 4500;
    z["Stutter"] = 4500;
    x["SerialNumber"] = 10;
    y["SerialNumber"] = 20;
    z["SerialNumber"] = 30;
    
    cout << "----------------------------------------------------" << endl;
    cout << "Beginning Isotropic Propagations... " << endl;
    cout << "----------------------------------------------------" << endl;
    
    TDSCF* x_p = new TDSCF(pr,x,Scratch);
    TDSCF* y_p = new TDSCF(pr,y,Scratch);
    TDSCF* z_p = new TDSCF(pr,z,Scratch);
    
    int samples = 1;
    while (true)
    {
        x_p->Propagate();
        y_p->Propagate();
        if (!z_p->Propagate())
            break;
    }
    delete x_p;
    delete y_p;
    delete z_p;
    delete Scratch;
};

// Debug Propagation with varied excitation fractions.
inline void CorrectDebugJob(rks* pr)
{
    std::map<std::string,double> NoField, xField, Rlx, Gs;
    TDSCF_Temporaries* Scratch = new TDSCF_Temporaries();
    
    // First the relaxation job.
    Rlx["ApplyImpulse"] = 0;
    Rlx["StartFromSetman"] = int(rem_read(REM_RTPPSTATE));
    Rlx["UpdateFockMatrix"] = 0;
    Rlx["SerialNumber"] = 1;
    Rlx["Stabilize"] = 2;
    Rlx["Decoherence"] = 1;
    Rlx["SaveDipoles"] = 0;
    Rlx["DipolesEvery"] = 1000;
    Rlx["StatusEvery"] = 2000;
    Rlx["SaveEvery"] = 5000;
    Rlx["SavePopulations"] = 1;
    Rlx["SavePopulationsEvery"] = 1000;
    Rlx["FourierEvery"] = 0;
    Rlx["WriteDensities"] = 0;
    Rlx["SaveFockEvery"] = 0;
    Rlx["TCLOn"] = 1;
    Rlx["MaxIter"] = 0; // Perform an infinite number of propagations.
    Rlx["SkipInitial"] = 1;
    
    NoField["ApplyImpulse"] = 0;
    NoField["Stabilize"] = 0;
    NoField["StartFromSetman"] = 0;
    NoField["Stutter"] = 0; // Pauses propagation every n iterations && does corrected fourier transform.
    NoField["SaveDipoles"] = 1;
    NoField["StatusEvery"] = 100;
    NoField["SaveEvery"] = 1000;
    NoField["WriteDensities"] = 0;
    NoField["FourierEvery"] = 0;
    NoField["SaveFockEnergies"] = 0;
    NoField["SaveFockEvery"] = 0;
    NoField["SerialNumber"] = 2;
    NoField["SkipInitial"] = 1;
    
    xField["Stutter"] = 0;
    xField["ApplyImpulse"] = 1;
    xField["Stabilize"] = 0;
    xField["StatusEvery"] = 100;
    xField["SaveEvery"] = 1000;
    xField["StartFromSetman"] = 0; // The final density of Rlx will be fed in.
    xField["SkipInitial"] = 1;
    
    Gs["ApplyImpulse"] = 1;
    Gs["Stabilize"] = 0;
    Gs["StartFromSetman"] = 0;
    Gs["Stutter"] = 0;
    Gs["SaveDipoles"] = 1;
    Gs["StatusEvery"] = 100;
    Gs["SaveEvery"] = 1000;
    Gs["WriteDensities"] = 0;
    Gs["FourierEvery"] = 0;
    Gs["SaveFockEnergies"] = 0;
    Gs["SaveFockEvery"] = 0;
    Gs["SerialNumber"] = 4;
    Gs["SkipInitial"] = 1;
    
    Rlx["MaxIter"] = (int)((1.0/0.024)/Rlx["dt"]); // 1 fs just to chill stuff out.
    TDSCF* Gs_p = new TDSCF(pr,Gs,Scratch);
    Gs_p->Propagate();
    
    int trl=0;
    for (double ef = 0.0; ef < 1.0; ef+= 0.1)
    {
        cout << "----------------------------------------------------" << endl;
        cout << "Beginning debug propagations... " << endl;
        cout << "Excited Fraction " << Gs_p->params["ExcitedFraction"] << endl;
        cout << "----------------------------------------------------" << endl;

        Rlx["ExcitedFraction"] = ef;
        TDSCF* Rlx_p = new TDSCF(pr,Rlx,Scratch);
        xField["SerialNumber"] = trl*100;
        ++trl;
        TDSCF* ToCorrect_p = new TDSCF(pr,xField,Scratch);
        TDSCF* NonStationary_p = new TDSCF(pr,NoField,Scratch);
        NonStationary_p->Sync(Rlx_p);
        ToCorrect_p->Sync(Rlx_p);
        NonStationary_p->Rho.diag().st().print("Initial Rho FFFFFF");
        ToCorrect_p->Rho.diag().st().print("Initial Rho FFFFFF");
        NonStationary_p->Propagate();
        ToCorrect_p->Propagate();
        ToCorrect_p->TransientAbsorptionCorrection(NonStationary_p,Gs_p);
        delete Rlx_p;
        delete NonStationary_p;
        delete ToCorrect_p;
    }
    delete Gs_p;
    delete Scratch;
};


#endif
