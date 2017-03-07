#ifndef TDSCF_H
#define TDSCF_H

// OMP
#ifdef _OPENMP
#if _OPENMP > 201300
#define HASUDR 1
#else
#define HASUDR 0
#endif
#else
#define HASUDR 0
#endif



#ifdef __cplusplus
  extern "C"
  {
#endif
int n, n2, n3, n4, n_aux,n_mo,n_occ, no,nv;
int tn, tn2, tn3, tn4;
mat RSinv;
mat H,S,X;
cx_mat F,V,C;
cx_mat rho0;
cube BpqR;
vec eigs;
double nuclear_energy, Ehf, Emp2;
double dt;

bool InCore;

cx_mat Vso;// Spin-orbital V.
cx_mat IntCur, VCur;
cx_mat cia, ciadot;
cx_double c0, c0dot;

//cx_mat Rho_, RhoM12_; // Not realy global shared variable. Just here to minimize the calling variable

std::complex<double> j(0.0,1.0);
double thresh = pow(10.0,-11.0);
    typedef std::complex<double> cx;
    // Include cpp function here as well


    void Initialize(double *h, double *s, double *x, double *b, std::complex<double> *f, int n_, int n_aux_,int nocc_, double Enuc, double dt);
    void InitFock(cx_mat& Rho,cx_mat& V_);
    void UpdateVi(const cx_mat& V_);


    void FieldOn(cx *r, double *nuc_, cx *mux_, cx *muy_, cx *muz_, double x, double y, double z, double Amp_, double Freq_, double tau_, double t0_, int Imp, int Cw);
    void InitMu(cx *muxo_, cx *muyo_, cx *muzo_, double *mu0_);


    void MMUT_step(std::complex<double> *r, std::complex<double> *rm12, double tnow);
    void Split_RK4_Step_MMUT(const arma::vec& eigval, const arma::cx_mat& Cu , const arma::cx_mat& oldrho, arma::cx_mat& newrho, const double tnow, const double dt);


    void TDTDA_step(cx *r, double tnow);
    void tdtda(cx_mat& Rho_, cx_mat& RhoDot_, const double time, cx_mat& hmu);

    void Call(cx* v, cx* c);
    bool cx_mat_equals(const arma::cx_mat& X,const arma::cx_mat& Y);
		bool delta(int p,int q);

    void BuildSpinOrbitalV();
		void BuildSpinOrbitalV2();
		int SpinOrbitalOffset(int i);

    void CIS();
    void HCIS();
    void CISD2(const double ee, cx_vec v);
    void CIS_step(cx_mat CIA, cx C0, cx_mat &ciadot, cx &c0dot, const double tnow);
    void CISDOT(cx_mat CIA, cx C0, cx_mat &ciadot, cx &c0dot, const double tnow, cx_mat& hmu);
    void Initialize2(double *h, double *s, double *x, double *b, std::complex<double> *f, int n_, int n_aux_,int nocc_, double Enuc, double dt_);

}
    //}
    cx_mat MakeRho(cx_mat Rho_, cx_mat cia, cx_double c0);
    cx_mat FockBuild(cx_mat& Rho, cx_mat& V_);
    cx RhoDot(const cx_mat& Rho_, cx_mat& RhoDot_, const double& time);
    cx_mat Delta2();
#ifdef __cplusplus

#endif



#endif
