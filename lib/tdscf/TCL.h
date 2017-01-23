#ifndef _TCL2_h
#define _TCL2_h

#include <iostream>
#include <complex>
#include <list>
#include <fstream>
#include <vector>
#include <map>
#include <complex>
#include <algorithm>

#ifdef MPI_VERSION
#include "mpi.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include <armadillo>
#include "math.h"

#include "qchem.h"
//#include "libgen.h"
#include "BasisSet.hh"
#include "MolecularBasis.h"
#include "AtomicBasis.h"
#include <libmdc/threading_policy.h>

#include "TCL_complex_functions.h"

using namespace std;
using namespace arma;
typedef complex<double> cdub;

static double FsPerAu = 0.0241888;

// Markov rates given a spectral density in the drude lorentz form.
// This performs \int_0^\inf dt \int_0^\inf dw J(w)
class GammaW
{
public:
    //
    double Beta;
	vector<double> PeakFreq;
	vector<double> PeakWidth;
	vector<double> PeakAmp;
    
    GammaW(const vector<double>& g,const vector<double>& l,const vector<double>& O,double Beta_arg): Beta(Beta_arg),PeakFreq(O), PeakWidth(g), PeakAmp(l)
    {
    }
    

    inline void LorentzMarkovRate(const complex<double>& Eps,const complex<double>& g,const complex<double>& l,const  complex<double>& O,const  complex<double>& Beta, complex<double>& gam) const
    {
        const static complex<double> j(0.0,1.0);
        const static complex<double> one(1.0,0.0);
        const static complex<double> two(2.0,0.0);
        const static complex<double> four(4.0,0.0);
        complex<double> Numerator(0.0,0.0);
        complex<double> Denom(0.0,0.0);
        complex<double> SecondTerm(0.0,0.0);
        complex<double> g2(0.0,0.0);
        g2 = pow(g,two);
        complex<double> betaO=Beta*O;
        
        Numerator = -two*j*l*(-two*exp(betaO)*(g2 + j*g*Eps + pow(O,two)) +(g2 + j*g*Eps + O*(-Eps + O))/exp(j*Beta*g) + exp(-j*Beta*g + two*betaO)*(g2 + j*g*Eps + O*(Eps + O)));
        Denom = (exp(-(j*Beta*g)) - exp(betaO))*(-one + exp(Beta*(-j*g + O)))*(pow(g + j*Eps,two) + pow(O,two));
        SecondTerm = (l*((four*j*g*Eps*(pow(g,two) + pow(Eps,two) + pow(O,two))*HarmonicNumber(((-j/two)*Beta*Eps)/M_PI))/(pow(pow(g,two) + pow(Eps,two),two) + two*(g - Eps)*(g + Eps)*pow(O,two) + pow(O,four)) - ((g - j*O)*HarmonicNumber(-(Beta*(g - j*O))/(two*M_PI)))/(g - j*(Eps + O)) + ((g - j*O)*HarmonicNumber((Beta*(g - j*O))/(two*M_PI)))/(g + j*(Eps - O)) - ((g + j*O)*HarmonicNumber(-(Beta*(g + j*O))/(two*M_PI)))/(g - j*Eps + j*O) + ((g + j*O)*HarmonicNumber((Beta*(g + j*O))/(two*M_PI)))/(g + j*(Eps + O))))/M_PI;
        gam += (Numerator/Denom + SecondTerm);
    };
    
    void GammaAtExact(const double& Eps, complex<double>& Gam) const
    {
        Gam *= 0.0;
        for (int kk =0 ; kk < PeakAmp.size(); ++kk)
		{
			LorentzMarkovRate(Eps,PeakWidth[kk],PeakAmp[kk],PeakFreq[kk],Beta, Gam);
		}
    }
    
    std::complex<double> GammaAtExact(const double& Eps) const
    {
        std::complex<double> Gam(0.0,0.0);
        for (int kk =0 ; kk < PeakAmp.size(); ++kk)
		{
			LorentzMarkovRate(Eps,PeakWidth[kk],PeakAmp[kk],PeakFreq[kk],Beta, Gam);
		}
        return Gam;
    }
    
    void GammaAtExactDb(const double& Eps, complex<double>& Gam) const
    {
        Gam *= 0.0;
        for (int kk =0 ; kk < PeakAmp.size(); ++kk)
		{
			LorentzMarkovRate(Eps,PeakWidth[kk],PeakAmp[kk],PeakFreq[kk],Beta, Gam);
            cout << PeakFreq[kk] << PeakAmp[kk] << PeakWidth[kk] << Gam << endl;
		}
    }
};

// Main class for electronic TCL dissipation.
// Besides the arguments, this depends on a file which contains J(w) in the DL form: SD.csv
class TCLMatrices
{
public:
	vector< vector<double> > PeakFreq;
	vector< vector<double> > PeakWidth;
	vector< vector<double> > PeakAmp;
    
    // Parameters
    std::map<std::string,double> params;
    bool low;// Use Lowdin fluctuations instead of MO.
	bool DiagonalHole;
    bool Decoherence;
	int No;
    int Nao;
    int n_thread;
    int n_e;
    int nvir;
	int Print;
    int UpdateFockMatrix;
	double Temp;
	double Beta; // 1/kbT in atomic units.
    double Thresh;
    bool Lamb;

    // intermediate quantities.
    arma::cx_cube GammaAtEpsilon;
    arma::cx_cube u;
    arma::cx_mat eps;
    arma::cx_vec fock;
    arma::cx_mat C;
    arma::cx_mat C_low;
    arma::cx_mat FastRates;
    
    vector<GammaW> Gis;
    vector<GammaW> Gis_low;
    
	TCLMatrices(int norb, int nao,  arma::cx_mat& C_, arma::vec & fock_, int noc, const std::map<std::string,double>& params_ = std::map<std::string,double>()) : C(C_), C_low(C_), eps(norb,norb), Gis(), Nao(nao), No(norb), GammaAtEpsilon(norb,norb,norb), n_e(noc), u(norb,norb,nao), low(false)
	{
        Temp = ((params_.find("Temp") != params_.end())? params_.find("Temp")->second : 300);
        DiagonalHole = ((params_.find("DiagonalHole") != params_.end())? params_.find("DiagonalHole")->second : false);
        Decoherence = ((params_.find("Decoherence") != params_.end())? params_.find("Decoherence")->second : true);
        Print = ((params_.find("Print") != params_.end())? params_.find("Print")->second : 0);
        Lamb = ((params_.find("Lamb") != params_.end())? params_.find("Lamb")->second : 1);
        Thresh = ((params_.find("Thresh") != params_.end())? params_.find("Thresh")->second : pow(10.0,-14.0));
        cout << "==============================================" << endl;
        cout << "=== Markovian Electronic TCL JAP 2014 ========" << endl;
        cout << "==============================================" << endl;
        n_thread = min(64,nao);
#ifdef _OPENMP
        n_thread = min(n_thread,omp_get_num_procs());
		omp_set_num_threads(n_thread);
#else
        n_thread = 1;
#endif
        cout << "Instantiating TCLMatrices, No:"  << No <<  ", Nao " << Nao << endl;
		cout << "n_e: " << n_e << " n_thread: " << n_thread << endl;
        cout << "C dimensions: " << C_.n_rows << "," << C_.n_cols << endl;
        cout << "==============================================" << endl;
        cout << "Decoherence: "  << Decoherence << endl;
        cout << "DiagonalHole: "  << DiagonalHole << endl;
        cout << "Lamb: "  << Lamb << endl;
		cout << "Print: "  << Print << endl;
        cout << "Thresh: " << Thresh << endl;
        UpdateFockMatrix=params["UpdateFockMatrix"];
        
        fock.resize(fock_.n_rows,fock_.n_cols);
        fock.zeros();
        fock.set_real(fock_);
        std::cout << std::setprecision(9) << scientific;
		double EvPerAu = 27.2113;
		double Kb = 8.61734315e-5/EvPerAu;
		double AuPerWavenumber = 4.5563e-6;
		Beta = 1.0/(Temp*Kb);
        cout << "Temp: " << Temp << endl;
        cout << "Beta: " << Beta << endl;
        cout << "==============================================" << endl;
        for (int a = 0; a<No; ++a)
        {
            for (int b = 0; b<No; ++b)
            {
                for (int m = 0 ; m<Nao; ++m)
                {
                    u(a,b,m) = conj(C(m,a))*C(m,b); // C is aoXmo (indices) moXao in memory.
                }
            }
        }
		
        fock.st().print("Eigenvalues:");
        for (int m = 0 ; m<No; ++m)
		{
			for (int a = 0; a<No; ++a)
			{
                eps(m,a) = fock[m]-fock[a];
			}
		}
        
        ReadJwFromDisk();
        {
            bool foundrates = false;
            try{
                FILE* file;
                foundrates = (file = fopen("markovrates", "r"));
                if (foundrates)
                {
                    fclose(file);
                    cx_cube ratestmp;
                    ratestmp.load("markovrates");
                    if(ratestmp.n_rows != No || ratestmp.n_cols != No || ratestmp.n_slices != Nao)
                    {
                        cout << "Wrong Dimension in Old Rates." << endl;
                        throw 1;
                    }
                    cout << "Loading precomputed rates from disk." << endl;
                    GammaAtEpsilon = arma::cx_cube(No,No,Nao);
                    GammaAtEpsilon.zeros();
                    foundrates = GammaAtEpsilon.load("markovrates");
                }
            }
            catch(...)
            { }
            if (!foundrates)
                rateUpdate();
        }
        
        if (!params["UpdateFockMatrix"] && (!params["Decoherence"]))
        {
            cout << "Building fast rate matrices..." << endl;
            BuildFastRates();
        }
        
        cout << "Redfield Rates have been built." << endl;
		TestGamma();
        cout << "TCL initialization is complete. " << endl;
        cout << "==============================================" << endl;
	}
	
	~TCLMatrices()
	{
	}
	
	void update(const cx_mat& C_, vec f_diag)
	{
        if (!is_finite(C_) || !is_finite(f_diag) )
		{
            cout << " TCL::update passed garbage" << endl;
            throw 1;
		}
        for (int a = 0; a<No; ++a)
        {
            for (int b = 0; b<No; ++b)
            {
                for (int m = 0 ; m<Nao; ++m)
                {
                    u(a,b,m) = conj(C_(m,a))*C_(m,b); // C is aoXmo (indices) moXao in memory.
                }
            }
        }
        fock.resize(f_diag.n_rows,f_diag.n_cols);
        fock.zeros();
		fock.set_real(f_diag);
        for (int m = 0 ; m<No; ++m)
		{
			for (int a = 0; a<No; ++a)
			{
                eps(m,a) = fock[m]-fock[a];
			}
		}
	}
    
    void rateUpdate()
	{
		if (Print)
			cout << "Updating the Markov rates. " << endl;
        {
            double tmp=0.0;
            GammaAtEpsilon = arma::cx_cube(No,No,Nao);
            GammaAtEpsilon.zeros();
#ifdef _OPENMP
		omp_set_num_threads(n_thread);
#endif
#pragma omp parallel for schedule(guided,2)
            for (int m = 0 ; m<Nao; ++m)
            {
                loadbar(m,No);
                for (int a = 0; a<No; ++a)
                {
                    for (int  n= 0; n<No; ++n)
                    {
                        
                        Gis[m].GammaAtExact(real(eps(a,n)),GammaAtEpsilon(a,n,m));
                        if (real(GammaAtEpsilon(a,n,m)) < 0.0)
                        {
                            if (abs(real(GammaAtEpsilon(a,n,m))) > pow(10.0,-7.0))
                                cout << " Warning... Negative real rate: " << real(GammaAtEpsilon(a,n,m)) << endl;
                            tmp = GammaAtEpsilon(a,n,m).imag();
                            GammaAtEpsilon(a,n,m)=std::complex<double>(0.0,tmp);
                        }
                    }
                }
            }
            GammaAtEpsilon.save("markovrates");
        }
        if (!is_finite(GammaAtEpsilon))
		{
            cout << " TCL::rateUpdate made garbage" << endl;
            throw 1;
		}
	}
	
    // Adjusts the chemical potential until N electrons are present in the system.
    cx_vec FermiDirac(int N,const cx_vec& ens)
    {
        cx_vec Tore(ens);
        typedef std::complex<double> cplx;
        cplx mu = ens(N);
        double Thresh = 0.0000001;
        cout << " Want " << N << " Fermions" << endl;
        //Tore.transform([=](cplx& ei){return 1/(1+exp(Beta*(ei-mu)));});    // This is only valid for C++11
        for (int i=0;i<ens.n_elem;++i)
            Tore(i) = 1.0/(1.0+exp(Beta*(ens(i)-mu)));
        while (true)
        {
            double su = real(sum(Tore));
            if ( su-N > Thresh)
            {
                mu -= Thresh/13.0;
            }
            else if ( (-1.0)*(su-N) > Thresh)
            {
                mu += Thresh/17.0;
            }
            else if (abs(su-N)<Thresh)
                break;
            Tore=ens;
            for (int i=0;i<ens.n_elem;++i)
                Tore(i) = 1.0/(1.0+exp(Beta*(ens(i)-mu)));
            //Tore.transform([=](cplx& ei){return 1/(1+exp(Beta*(ei-mu)));});    // This is only valid for C++11
        }
        cout << " Populated with " << sum(Tore) << " Fermions at mu =" << mu << endl;
        return Tore;
    }
    
    // Test the markov rate
    // Print out equilbrium populations.
    void TestGamma(void)
	{
        std::vector<double> amp,freq,gam;
        amp.push_back(0.001);
        freq.push_back(0.0001);
        gam.push_back(0.000005);
        GammaW tmp(gam,amp,freq,Beta);
        std::complex<double> tmp1(0.0,0.0), tmp2(0.0,0.0);
        tmp.GammaAtExact(0.001,tmp1);
        tmp.GammaAtExact(-0.001,tmp2);
        cout << " TestGamma: " << tmp1 << tmp2 << endl;
		cout << " Lumo-Homo Frequency: " << eps(n_e,n_e-1) << endl;
		cout << " Lumo->Homo timescale (unblocked, au): " << 1.0/Relement(n_e-1,n_e-1,n_e,n_e) << endl;
		cout << " Homo->Lumo timescale (unblocked, au): " << 1.0/Relement(n_e,n_e,n_e-1,n_e-1) << endl;
        if (Decoherence)
            cout << " Homo,Lumo dephasing timescale (unblocked, au): " << 1.0/Relement(n_e,n_e-1,n_e,n_e-1) << endl;
		cout << " *** IF THE ABOVE ARE CLOSE TO dt you should expect the dynamics to become chaotic *** " << endl;
    }
    
	void ReadJwFromDisk(void)
	{
		double EvPerAu = 27.2113;
		double Kb = 8.61734315e-5/EvPerAu;
		double AuPerWavenumber = 4.5563e-6;
		{
            ifstream reader;
            cout << " Looking for SD.csv, && Drude parameters in atomic units. " << endl;
            if (low)
                reader.open("./SD_low.csv");
            else
                reader.open("./SD.csv");
            reader.seekg(0);
            if(reader.is_open())
            {
                while(!reader.eof())
                {
                    int npeaks,norb;
                    PeakFreq.push_back(vector<double>() );
                    PeakWidth.push_back(vector<double>() );
                    PeakAmp.push_back(vector<double>() );
                    reader >> norb;
                    norb--; // This should be fixed in the matlab.
                    if (norb < 0)
                    {
                        int tmp;
                        reader >> tmp;
                        cout << "Error, negative orbital " << norb << " " << tmp <<  endl;
                        throw;
                    }
                    reader >> npeaks;
                    for(int a = 0; a < npeaks; a++)
                    {
                        double x, y, z;
                        // Order in the matlab.
                        reader >> x >> y >> z;
                        double O = x; // units: energy.
                        double l = y; // units: energy.
                        double g = z; // units: energy.
                        // Frequencies should never be greater than one.
                        if (O>1.)
                        {
                            O*=AuPerWavenumber;
                            g*=AuPerWavenumber;
                            l*=AuPerWavenumber;
                        }
                        if (O/AuPerWavenumber > 6000.0)
                        {
                            cout << "Very high frequency oscillator: " << O << " " << O/AuPerWavenumber << endl;
                        }
                        if (l>1.0)
                        {
                            cout << "Suspiciously strong bath." << endl;
                            cout << "Orb: " << norb << " Omega: " <<  O << " Gamma: " << g << " lambda: " << l << endl;
                            l=0.0;
                        }
                        if (l >= 0.0)
                        {
                            PeakWidth[norb].push_back(g);
                            PeakAmp[norb].push_back(l);
                            PeakFreq[norb].push_back(O);
                        }
                    }
                    if (norb >= Nao-1)
                        break;
                }
            }
            else
            {
                cout << "Didn't find SD.csv" << endl;
            }
            
            reader.close();
            
            if (PeakAmp.size() == 0)
            {
                cout << "Read no peaks !*!*!*!*!!*!*!*!*!*!*!*!*!!*!*!*!*!*!*!*!!*!*!*" << endl;
                cout << "Read no peaks !*!*!*!*!!*!*!*!*!*!*!*!*!!*!*!*!*!*!*!*!!*!*!*" << endl;
                for (int i=0; i<Nao; ++i)
                {
                    PeakFreq.push_back(vector<double>() );
                    PeakWidth.push_back(vector<double>() );
                    PeakAmp.push_back(vector<double>() );
                    PeakWidth[i].push_back(0.00001);
                    PeakAmp[i].push_back(0.0001);
                    PeakFreq[i].push_back(0.0001);
                    PeakWidth[i].push_back(0.001);
                    PeakAmp[i].push_back(0.0017);
                    PeakFreq[i].push_back(0.0017);
                    PeakWidth[i].push_back(0.01);
                    PeakAmp[i].push_back(0.017);
                    PeakFreq[i].push_back(0.007);
					PeakWidth[i].push_back(0.01);
                    PeakAmp[i].push_back(0.010);
                    PeakFreq[i].push_back(0.009);
                }
            }
            else
            {
                cout << "PeakWidth.size(): " << PeakWidth.size() << endl;
                cout << "PeakAmp.size(): " << PeakAmp.size() << endl;
                cout << "PeakFreq.size(): " << PeakFreq.size() << endl;
            }
            cout << "Creating Markov rates..." << endl;
            for (int i=0; i<Nao ; ++i)
                Gis.push_back(GammaW(PeakWidth[i],PeakAmp[i],PeakFreq[i],Beta));
            return;
        }
	}
    
	void ContractGammaRhoMThreads(arma::cx_mat& OutRho,const arma::cx_mat& Rho) const
	{
		int n = Rho.n_cols;
        if (!Decoherence)
        {
            ContractGammaRhoMThreads_diag(OutRho, Rho);
            return;
        }
        
        threading_policy::enable_omp_only();
		if (n != No)
		{
			cout << "Argument Error...";
			throw;
		}
        cx_mat HoleRho(Rho.n_rows,Rho.n_cols);
		HoleRho.eye();
		HoleRho = HoleRho - Rho;
		if (DiagonalHole)
			diagmat(HoleRho);
		for (int i=0; i<n ; ++i)
		{
			// This should never occur... but we get rounding errors sometimes.
			if (real(Rho(i,i))>1.0)
				HoleRho(i,i)=0.0;
		}
		
		{
			arma::cx_cube i1(n,n,n_thread);
			arma::cx_cube i2(n,n,n_thread);
            arma::cx_cube tmp(n,n,n_thread);
			i1.zeros(); i2.zeros();	tmp.zeros();
			cx_mat um(n,n);
			cx_mat uG(n,n);
			cx_mat uGst(n,n);
			cx_mat uG_d(n,n);
			cx_mat uGst_nd(n,n);
			cx_mat uhr(n,n);
			cx_mat hru(n,n);
#ifdef _OPENMP
		omp_set_num_threads(n_thread);
#endif
#pragma omp parallel for private(um,uG,uGst,uG_d,uGst_nd,uhr,hru) shared(i1,i2,tmp) schedule(static)
			for (int m = 0; m<Nao; ++m)
			{
				// Useful intermediates that must be recomputed.
				um = u.slice(m);
                uG = um%(GammaAtEpsilon.slice(m));
                uGst = um%((GammaAtEpsilon.slice(m)).st());
                if (!Lamb)
                {
                    uG.set_imag(zeros<mat>(uG.n_rows,uG.n_cols));
                    uGst.set_imag(zeros<mat>(uGst.n_rows,uGst.n_cols));
                }
				uG_d = diagmat(uG);
				uGst_nd = uGst - diagmat(uGst);
                uhr = um*HoleRho;
				hru = HoleRho*(um.t());
                
#ifdef _OPENMP
				i1.slice(omp_get_thread_num()) += diagmat(um*HoleRho)*uG_d;
				i1.slice(omp_get_thread_num()) += diagmat(um*HoleRho*uGst_nd); // Term1
				i2.slice(omp_get_thread_num()) += (diagmat(conj(um)*HoleRho)%conj(uG_d));
				i2.slice(omp_get_thread_num()) += diagmat(conj(um)*HoleRho*conj(uGst_nd)); //Term2
				tmp.slice(omp_get_thread_num()) += uG_d*Rho*diagmat(uhr);
				tmp.slice(omp_get_thread_num()) += diagmat( uGst_nd*diagmat(Rho)*(uhr - diagmat(uhr)));
				tmp.slice(omp_get_thread_num()) += diagmat(hru)*Rho*conj(uG_d);
				tmp.slice(omp_get_thread_num()) += diagmat( (hru-diagmat(hru))*diagmat(Rho)*(uGst_nd.t()) );
#else
				i1.slice(0) += diagmat(um*HoleRho)*uG_d;
				i1.slice(0) += diagmat(um*HoleRho*uGst_nd); // Term1
				i2.slice(0) += (diagmat(conj(um)*HoleRho)%conj(uG_d));
				i2.slice(0) += diagmat(conj(um)*HoleRho*conj(uGst_nd)); //Term2
				tmp.slice(0) += uG_d*Rho*diagmat(uhr); // Term3
				tmp.slice(0) += diagmat( uGst_nd*diagmat(Rho)*(uhr - diagmat(uhr)));
				tmp.slice(0) += diagmat(hru)*Rho*conj(uG_d);
				tmp.slice(0) += diagmat( (hru-diagmat(hru))*diagmat(Rho)*(uGst_nd.t()) ); // Term 4
#endif
                
			}
            
			if (n_thread>1)
            {
                for (int m = 1; m<n_thread; ++m)
                {
                    i1.slice(0) += i1.slice(m);
                    i2.slice(0) += i2.slice(m);
                }
            }
            for (int m = 0; m<n_thread; ++m)
				OutRho += tmp.slice(m);

            OutRho -= i1.slice(0)*Rho;
			OutRho -= Rho*i2.slice(0);
		}
        threading_policy::enable_omp_only();
	}

    // Only population relaxation, enormous savings.
    void ContractGammaRhoMThreads_diag(arma::cx_mat& OutRho,const arma::cx_mat& Rho_) const
	{
        if (!UpdateFockMatrix)
        {
            ContractGammaRhoMThreads_FAST(OutRho,Rho_);
            return;
        }
        
        cx_mat Rho = diagmat(Rho_);
        threading_policy::enable_omp_only();
		int n = Rho.n_cols;
		if (n != No)
		{
			cout << "Argument Error...";
			throw;
		}
		
        cx_mat OutRhod(Rho.n_rows,1); OutRhod.zeros();
		cx_mat HoleRho(Rho.n_rows,Rho.n_cols);
		HoleRho.eye();
		HoleRho = HoleRho - Rho;
        
        HoleRho=diagmat(HoleRho);

        for (int i=0; i<n ; ++i)
		{
			// This should never occur... but we get rounding errors sometimes.
			if (real(Rho(i,i))>1.0)
				HoleRho(i,i)=0.0;
		}
        
        
        cx_mat HoleRhod(Rho.n_rows,1);
        cx_mat Rhod(Rho.n_rows,1);
        HoleRhod.col(0) = HoleRho.diag();
        Rhod.col(0) = Rho.diag();
		
		{
			arma::cx_mat i1(n,n_thread);
			arma::cx_mat i2(n,n_thread);
            arma::cx_mat tmp(n,n_thread);
			i1.zeros(); i2.zeros();	tmp.zeros();
#ifdef _OPENMP
            omp_set_num_threads(n_thread);
#endif
#pragma omp parallel for shared(i1,i2,tmp,Rho,HoleRho,Rhod,HoleRhod) schedule(guided)
            for (int m = 0; m<Nao; ++m)
			{
                cx_mat um(n,n); um.zeros();
                cx_mat uG(n,n); uG.zeros();
                cx_mat uGst(n,n); uGst.zeros();
                cx_mat uG_d(n,1); uG_d.zeros();
                cx_mat uGst_nd(n,n); uGst_nd.zeros();
				// Useful intermediates that must be recomputed.
				um = u.slice(m);
                uG = um%(GammaAtEpsilon.slice(m));
                uGst = um%((GammaAtEpsilon.slice(m)).st());
				uG_d.col(0) = uG.diag();
				uGst_nd = uGst - diagmat(uGst);
                
#ifdef _OPENMP
                cx_mat umhr = (um*HoleRho);
                i1.col(omp_get_thread_num()) += (umhr.diag())%uG_d;
                cx_mat umhrugstnd = (um*HoleRho*uGst_nd);
                i1.col(omp_get_thread_num()) += umhrugstnd.diag();
                
                cx_mat cumhr = conj(um)*HoleRho;
                i2.col(omp_get_thread_num()) += (cumhr.diag())%conj(uG_d);
                cx_mat cumhrugstnd = conj(um)*HoleRho*conj(uGst_nd);
				i2.col(omp_get_thread_num()) += cumhrugstnd.diag();
                
                cx_mat uhr(n,n);
                uhr = um*HoleRho;
                tmp.col(omp_get_thread_num()) += uG_d%Rhod%(uhr.diag());
                cx_mat ugstndruhrnd = ( uGst_nd*diagmat(Rho)*(uhr - diagmat(uhr)));
                tmp.col(omp_get_thread_num()) += ugstndruhrnd.diag();
                
				cx_mat hru = HoleRho*(um.t());
                tmp.col(omp_get_thread_num()) += (hru.diag())%Rhod%conj(uG_d);
                cx_mat hrundugstnd = (hru-diagmat(hru))*diagmat(Rho)*(uGst_nd.t());
                tmp.col(omp_get_thread_num()) +=(hrundugstnd.diag());
#else
                cx_mat umhr = (um*HoleRho);
                i1.col(0) += (umhr.diag())%uG_d;
                cx_mat umhrugstnd = (um*HoleRho*uGst_nd);
                i1.col(0) += umhrugstnd.diag();

                cx_mat cumhr = conj(um)*HoleRho;
                i2.col(0) += (cumhr.diag())%conj(uG_d);
                cx_mat cumhrugstnd = conj(um)*HoleRho*conj(uGst_nd);
				i2.col(0) += cumhrugstnd.diag();

                cx_mat uhr(n,n);
                uhr = um*HoleRho;
                tmp.col(0) += uG_d%Rhod%(uhr.diag());
                cx_mat ugstndruhrnd = ( uGst_nd*diagmat(Rho)*(uhr - diagmat(uhr)));
                tmp.col(0) += ugstndruhrnd.diag();

				cx_mat hru = HoleRho*(um.t());
                tmp.col(0) += (hru.diag())%Rhod%conj(uG_d);
                cx_mat hrundugstnd = (hru-diagmat(hru))*diagmat(Rho)*(uGst_nd.t());
                tmp.col(0) += (hrundugstnd.diag());
#endif
            }
            
			if (n_thread>1)
            {
                for (int m = 1; m<n_thread; ++m)
                {
                    i1.col(0) += i1.col(m);
                    i2.col(0) += i2.col(m);
                }
            }
            
            for (int m = 0; m<n_thread; ++m)
				OutRhod.col(0) += tmp.col(m);
            
			OutRhod -= (i1.col(0)%Rhod);
			OutRhod -= (Rhod%i2.col(0));
            OutRho.diag() = OutRhod.col(0);

		}
        threading_policy::enable_omp_only();
	}
    
    // Only population relaxation, enormous savings.
    // No loop over m indices.
    void BuildFastRates()
    {
        threading_policy::enable_omp_only();
        int n=No;
        FastRates.resize(No,No); FastRates.zeros();
        {
#ifdef _OPENMP
            omp_set_num_threads(n_thread);
#endif

            for (int m = 0; m<Nao; ++m)
            {
                cx_mat um(n,n); um.zeros();
                cx_mat uGst(n,n); uGst.zeros();
                // Useful intermediates that must be recomputed.
                um = u.slice(m);
                uGst = um%((GammaAtEpsilon.slice(m)).st());
#pragma omp parallel for
                for(int p=0; p<No; ++p)
                for(int q=0; q<No; ++q)
                    FastRates(p,q) += um(p,q)*uGst(q,p);
            }
        }
        threading_policy::enable_omp_only();
    }
    
    // Only the diagonal...
    void ContractGammaRhoMThreads_FAST(arma::cx_mat& OutRho,const arma::cx_mat& Rho_) const
    {
        cx_vec rd = Rho_.diag();
        cx_vec one(No); one.ones();
        cx_vec ed = one-rd;
        cx_vec od(No); od.zeros();

#pragma omp parallel for
        for (int a=0; a<No; ++a)
        {
            for (int c=0; c<No; ++c)
            {
                od(a) += FastRates(a,c)*rd(a)*ed(c);
                od(a) += conj(FastRates(a,c))*rd(a)*ed(c);

                od(a) -= FastRates(c,a)*rd(c)*ed(a);
                od(a) -= conj(FastRates(c,a))*rd(c)*ed(a);
            }
        }
        OutRho -= diagmat(od);
        threading_policy::enable_omp_only();
    }
    
    // Allows one to calculate the Redfield tensor elementwise.
    std::complex<double> Relement(int p, int q, int r, int s)
    {
		if (p>=No || q>=No || r>=No || s>=No || p<0 || q<0 || r<0 || s<0)
		{
			cout << "Relement argument error..." << p << q << r << s << endl;
			return 0.0;
		}
        arma::cx_mat Rho(No,No);
        arma::cx_mat OutRho(No,No);
        Rho.zeros();
        OutRho.zeros();
        Rho(r,s) = 1.0;
		rateUpdate(); // Otherwise the rates may not be made.
        ContractGammaRhoMThreads(OutRho,Rho);
        return OutRho(p,q);
    }
};


#endif
