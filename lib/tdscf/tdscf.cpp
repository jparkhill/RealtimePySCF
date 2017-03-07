#include <iostream>
#include <armadillo>
using namespace arma;
#include "tdscf.h"
#include "field.h"
#include "bo.h"
//#include "tddft.h"
#include "func.h"

void Initialize(double *h, double *s, double *x, double *b, std::complex<double> *f, int n_, int n_aux_,int nocc_, double Enuc, double dt_)
{
  // save the variables that does not change over the dynamics
  // make n and naux public
  // H(AO), B(LAO), V is initialized as eye.
  // make empty eigs vector
  n_mo = n = n_;
  n2 = n*n;
  n3 = n2*n;
  n4 = n3*n;
  tn=2*n_mo;
  tn2=4*n_mo*n_mo;
  tn3=8*n2*n_mo;
  tn4=16*n3*n_mo;
  n_aux = n_aux_;
  no = n_occ = nocc_;
  nv = n_mo-n_occ;
  dt = dt_;
  nuclear_energy = Enuc;
  InCore = (n*n*n*n*sizeof(std::complex<double>)/(1024*1024) < 10.0);
  IntCur.resize(n*n,n*n);

  mat Htmp(h,n,n);
  mat Stmp(s,n,n);
  mat Xtmp(x,n,n);
  cube BpqRtmp(b,n_aux,n,n);
  H = Htmp; S = Stmp; X = Xtmp;
  BpqR.resize(n,n,n_aux);
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
      for(int k = 0; k < n_aux; k++)
        BpqR(i,j,k) = BpqRtmp(k,i,j);
  // Transform BpqR(AO,AO,naux) to BpqR(LAO,LAO,naux)
  for(int R = 0; R < n_aux; R++)
  {
    mat B1 = X.t() * BpqR.slice(R) * X;
    BpqR.slice(R) = B1;
  }
  cout << "H,S,X,BpqR Transferred..." << endl;

  V.resize(n,n); V.eye(); cx_mat V_ = V;
  eigs.resize(n); eigs.zeros();


  cout << "Initial guess" << endl;
  cx_mat Rho_(n,n); Rho_.eye();
  Rho_.submat(n_occ,n_occ,n_mo-1,n_mo-1) *= 0.0;
  FockBuild(Rho_,V_);
  UpdateVi(V_);

  InitFock(Rho_,V_);

  memcpy(f,F.memptr(),2*sizeof(double)*n*n);

}

void Initialize2(double *h, double *s, double *x, double *b, std::complex<double> *f, int n_, int n_aux_,int nocc_, double Enuc, double dt_)
{
  // save the variables that does not change over the dynamics
  // make n and naux public
  // H(AO), B(LAO), V is initialized as eye.
  // make empty eigs vector
  n_mo = n = n_;
  n2 = n*n;
  n3 = n2*n;
  n4 = n3*n;
  tn=2*n_mo;
  tn2=4*n_mo*n_mo;
  tn3=8*n2*n_mo;
  tn4=16*n3*n_mo;
  n_aux = n_aux_;
  no = n_occ = nocc_;
  nv = n_mo-n_occ;
  dt = dt_;
  nuclear_energy = Enuc;
  InCore = (n*n*n*n*sizeof(std::complex<double>)/(1024*1024) < 10.0);
  IntCur.resize(n*n,n*n);

  mat Htmp(h,n,n);
  mat Stmp(s,n,n);
  mat Xtmp(x,n,n);
	cx_mat cia;
  cube BpqRtmp(b,n_aux,n,n);
  H = Htmp; S = Stmp; X = Xtmp;
  BpqR.resize(n,n,n_aux);
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
      for(int k = 0; k < n_aux; k++)
        BpqR(i,j,k) = BpqRtmp(k,i,j);
  // Transform BpqR(AO,AO,naux) to BpqR(LAO,LAO,naux)
  for(int R = 0; R < n_aux; R++)
  {
    mat B1 = X.t() * BpqR.slice(R) * X;
    BpqR.slice(R) = B1;
  }
  cout << "H,S,X,BpqR Transferred..." << endl;

  V.resize(n,n); V.eye(); cx_mat V_ = V;
  eigs.resize(n); eigs.zeros();


  cout << "Initial guess" << endl;
  cx_mat Rho_(n,n); Rho_.eye();
  Rho_.submat(n_occ,n_occ,n_mo-1,n_mo-1) *= 0.0;
  FockBuild(Rho_,V_);
  UpdateVi(V_);
	CIS();
	HCIS();
	cia.resize(no,nv);
	c0 = 1.0;
  InitializeExpectation(MakeRho(Rho_,cia,c0));
  InitFock(Rho_,V_);

  memcpy(f,F.memptr(),2*sizeof(double)*n*n);

}

void InitFock(cx_mat& Rho,cx_mat& V_)
{
  //setup the matrix from arrays

  cout << "Solving the RI-HF problem to ensure self-consistency." << endl;
  double err=100.0;
  int maxit = 400; int it=1;
  rho0 = V*Rho*(V.t());
  //rho0.print("Inintial Rho");

  while (std::abs(err) > pow(10.0,-10) && it<maxit)
  {
    cx_mat Rot = FockBuild(Rho,V_);
    Rho = Rot.t()*Rho*Rot;
    Rho.diag() *= 0.0;
    err=accu(abs(Rho));
    Rho.eye();
    Rho.submat(n_occ,n_occ,n_mo-1,n_mo-1)*=0.0;
    if (it%10==0)
      cout << " Fock-Roothan: " << it  << " Ehf" << Ehf << endl;
    it++;
  }
  FockBuild(Rho,V_);
  UpdateVi(V_);

  eigs.t().print("EigenValues");

}

cx_mat FockBuild(cx_mat& Rho, cx_mat& V_)
{

  mat Xt = X.t(); // X(AO x LAO)
  cx_mat Vt = V.t(); // V (LAO x MO)
  F.resize(Rho.n_rows,Rho.n_cols); F.zeros();
  mat J(Rho.n_rows,Rho.n_cols); J.zeros();
  cx_mat K(Rho); K.zeros();
  cx_mat Rhol = V*Rho*Vt; // Get Rho into the lowdin basis for the fock build.
  mat h = 2.0*(Xt)*(H)*X; // H is stored in the ao basis. h(LAO)
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
  //F(LAO)
  F.set_real(J+0.5*h);
  //F += 0.5*h;
  //F += J;
  F += K;

  F = Vt*F*V;    // Move it back into the Prev eigenbasis.
  cx_mat Jx = Vt*J*V;    // Move it back into the Prev eigenbasis.
  cx_mat hx = Vt*h*V;    // Move it back into the Prev eigenbasis.
  K = Vt*K*V;    // Move it back into the Prev eigenbasis.


  // F(MO)
  F = 0.5*(F+F.t());

  double Eone= real(trace(Rho*hx));
  double EJ= real(trace(Rho*Jx));
  double EK= real(trace(Rho*K));
  // debug
  //double EF = real(trace(Rho*F));
  //double EFF = EF + nuclear_energy;
  //cout << "FOCK ENERGY:" << EFF << endl;
  Ehf = Eone + EJ + EK + nuclear_energy;

  cx_mat Vprime; eigs.zeros();
  if (!eig_sym(eigs, Vprime, F))
  {
    cout << "eig_sym failed to diagonalize fock matrix ... " << endl;
    throw 1;
  }

  //eigs.print("eigs");
  V_ = V = V*Vprime;
  C = X*V;

  // UPDATE USING MEMCPY
  // F, V_, V, C, eigs are updated




  return Vprime;
}

void FieldOn(cx *r, double *nuc_, cx *mux_, cx *muy_, cx *muz_, double x, double y, double z, double Amp_, double Freq_, double tau_, double t0_, int Imp, int Cw)
{

  cx_mat Rho_(r,n,n);

  // mu
  vec nuc(nuc_,3);
  mu_n = nuc;
  cx_mat xtmp(mux_,n,n);
  cx_mat ytmp(muy_,n,n);
  cx_mat ztmp(muz_,n,n);

  mux = xtmp; muy = ytmp; muz = ztmp;
  pol[0] = x; pol[1] = y; pol[2] = z;

  // parameters
  Amp = Amp_;
  Freq = Freq_;
  tau = tau_;
  t0 = t0_;
  ApplyImpulse = Imp;
  ApplyCw = Cw;

  updatefield(C); // muo are prepared
  InitializeExpectation(Rho_);


}

void InitMu(cx *muxo_, cx *muyo_, cx *muzo_, double *mu0_)
{
  cx_mat xt = muxo.t();
  cx_mat yt = muyo.t();
  cx_mat zt = muzo.t();

  memcpy(muxo_,xt.memptr(),2*sizeof(double)*n*n);
  memcpy(muyo_,yt.memptr(),2*sizeof(double)*n*n);
  memcpy(muzo_,zt.memptr(),2*sizeof(double)*n*n);
  memcpy(mu0_,mu_0.memptr(),sizeof(double)*3);

}


void MMUT_step(std::complex<double> *r, std::complex<double> *rm12, double tnow)
{
  cx_mat Rho_(r,n,n);
  cx_mat RhoM12_(r,n,n);
  //cx_mat Rho_ = rtmp; cx_mat RhoM12_ = rm12tmp;
  // need Rho, Rhom12, tnow,
  vec Gd; cx_mat Cu;
  eig_sym(Gd,Cu,F);
  //Gd.print("Eigval");
  //Cu.print("Eigvec");
  //F.print("F with Field(MO)");
  //Rho_.print("Rho"); //MO
  //cx_mat Fao = C*F*C.t();
  //Fao.print("Fao");
  //C.print("C");
  //C.t().print("Ct");
  // Full step RhoM12 to make new RhoM12.
  cx_mat NewRhoM12(Rho_);
  cx_mat NewRho(Rho_);
  Split_RK4_Step_MMUT(Gd, Cu, RhoM12_, NewRhoM12, tnow, dt);
  Split_RK4_Step_MMUT(Gd, Cu, NewRhoM12, NewRho, tnow, dt/2.0);
  Rho_ = 0.5*(NewRho+NewRho.t());
  RhoM12_ = 0.5*(NewRhoM12+NewRhoM12.t());

  cx_mat rt = Rho_.t();
  cx_mat rm12t = RhoM12_.t();
  memcpy(r,rt.memptr(),2*sizeof(double)*n*n);
  memcpy(rm12,rm12t.memptr(),2*sizeof(double)*n*n);
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


void TDTDA_step(cx *r, double tnow)
{
  //cx_mat rtmp(r,n,n);
  cx_mat Rho_(r,n,n);
  Rho_ = Rho_.t();

  mat Xt = X.t();
  cx_mat Vt = V.t();
  cx_mat hmu = Vt*(Xt)*(H)*X*V;
  cx_mat dRho_(Rho_);

  BuildSpinOrbitalV();

  cx_mat newrho(Rho_); newrho.zeros();
  newrho = Rho_;

  // RK4
  cx_mat k1(newrho),k2(newrho),k3(newrho),k4(newrho),v2(newrho),v3(newrho),v4(newrho);
  k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
  v2.zeros(); v3.zeros(); v4.zeros();

  tdtda( Rho_, k1,tnow,hmu);
  v2 = (dt/2.0) * k1;
  v2 += Rho_;

  tdtda(  v2, k2,tnow+(dt/2.0),hmu);
  v3 = (dt/2.0) * k2;
  v3 += Rho_;

  tdtda(  v3, k3,tnow+(dt/2.0),hmu);
  v4 = (dt) * k3;
  v4 += Rho_;

  tdtda(  v4, k4,tnow+dt,hmu);
  cx_mat tmp(k1); tmp.zeros();
  tmp = dt*(1.0/6.0)*k1 + dt*(2.0/6.0)*k2 + dt*(2.0/6.0)*k3 + dt*(1.0/6.0)*k4;
  newrho += tmp;


  Rho_ = newrho;
  cx_mat rt = Rho_.t();
  memcpy(r,rt.memptr(),2*sizeof(double)*n*n);
}

void tdtda(cx_mat& Rho_, cx_mat& RhoDot_, const double tnow, cx_mat& hmu)
{
  cx_mat h = hmu;
  ApplyField(h,tnow);

#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
#endif
  const cx* Vi = IntCur.memptr();
  cx_mat tmp(RhoDot_); tmp.zeros();
  cx_mat J(RhoDot_); J.zeros();
  cx_mat K(RhoDot_); K.zeros();

  // Set up Rho(t) and Rho(0)
  cx_mat Rho0(Rho_); Rho0.eye();
  Rho0.submat(no,no,n-1,n-1).zeros();
  cx_mat Rho1(Rho_);


  RhoDot_+= -j * (h*Rho1 - Rho1*h);


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
  tmp=(J+K)*Rho1 - Rho1*(J+K);
  RhoDot_ +=  tmp;
  tmp.zeros();
  J.zeros(); K.zeros();

#if HASUDR
#pragma omp parallel for reduction(+: J,K) schedule(static)
#endif
  for (int ip=0; ip<n; ++ip)
      for (int i=0; i<n; ++i)
      {
          for (int ap=0; ap<n; ++ap)
              for (int k=0; k<n; ++k)
              {
                  J[ap*n+ip] += -j*2.0*(Vi[ap*n3+i*n2+ip*n+k])*Rho1[k*n+i];
                  K[ap*n+ip] -= -j*(Vi[ap*n3+i*n2+k*n+ip])*Rho1[k*n+i];
              }
      }
  tmp=(J+K)*Rho0 - Rho0*(J+K);
  RhoDot_ +=  tmp;

  // Debugging? Updating Fock to calculate Energy
  //F = 1.0*h + J + K ;
}

void Call(cx* v, cx* c)
{

  cx_mat vt = V.t();
  memcpy(v,vt.memptr(),2*sizeof(double)*n*n);
  cx_mat ct = C.t();
  memcpy(c,ct.memptr(),2*sizeof(double)*n*n);
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

								}
}

cx RhoDot(const cx_mat& Rho_, cx_mat& RhoDot_, const double& time)
{
	return 0.0;
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
            std::complex<double> A = B(p,q);
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
            Emp2 += real(IntCur[i*n3+j*n2+a*n+b]*(2.0*IntCur[i*n3+j*n2+a*n+b] - IntCur[i*n3+j*n2+b*n+a])/(d2(a,i)+d2(b,j)));
          }
        }
      }
    }
    cout <<  "Mp2 Correlation energy: " << Emp2 << endl;

    VCur = V_;
}


bool cx_mat_equals(const arma::cx_mat& X,const arma::cx_mat& Y)
{
  double tol=pow(10.0,-12.0);
  if (X.n_rows != Y.n_rows)
  return false;
  if (X.n_cols != Y.n_cols)
  return false;
  bool close(false);
  if(arma::max(arma::max(arma::abs(X-Y))) < tol)
  close = true;
  return close;
}


 bool delta(int p,int q)
 {
 		if (p==q)
 				return true;
 		return false;
 }

cx_mat Delta2()
{
  bool Sign=true;
  if (eigs.n_elem == 0)
  cout << "Err.." << endl;
  cx_mat tore(eigs.n_elem,eigs.n_elem);
  for(int i=0; i<eigs.n_elem; ++i)
  for(int j=0; j<eigs.n_elem; ++j)
  tore(i,j) = (Sign)? (eigs(i) - eigs(j)) : eigs(i) + eigs(j);
  tore.diag().zeros();
  return tore;
}

void CIS()
{
		const cx* Vi = IntCur.memptr();

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
		eigval.st().print("CIS eigval: ");

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
				Rho0.diag().st().print("Diag Pops: ");

				//debug CISD here:
				CISD2(eigval(ii) , v);
		}
		return;
}

void CISD2(const double ee, cx_vec v)
{  //CISD naive version, check energy only.
	 BuildSpinOrbitalV2();
	 // Make a spin orbital version of the amplitude too.
	 cx_mat bai(2*nv,2*no); bai.zeros();
	 for (int i=0; i<2*no; ++i)
			 for (int a=0; a<2*nv; ++a)
			 {
					 // Only singlets.
					 if ( ((i < no) && !(a<nv)) || (!(i < no) && (a<nv)) )
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
/*
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
													 e2 -= 0.25*(-1.0)*real(bai(a,i)*bai(b,j)*(Vi[(d+tno)*tn3+(e+tno)*tn2+(a+tno)*tn+j]-Vi[(d+tno)*tn3+(e+tno)*tn2+j*tn+(a+tno)])*(Vi[(d+tno)*tn3+(e+tno)*tn2+(b+tno)*tn+i]-Vi[(d+tno)*tn3+(e+tno)*tn2+i*tn+(b+tno)])/(eigs((d%nv)+no)+eigs((e%nv)+no)-eigs((j%no))-eigs((i%no)) - ee/27.2113));

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
													 e7 -= 0.25*(-1.0)*real(bai(a,i)*bai(b,j)*(Vi[i*tn3+(d+tno)*tn2+l*tn+j]-Vi[i*tn3+(d+tno)*tn2+j*tn+l])*(Vi[(d+tno)*tn3+(a+tno)*tn2+(b+tno)*tn+l]-Vi[(d+tno)*tn3+(a+tno)*tn2+l*tn+(b+tno)])/(eigs((d%nv)+no)+eigs((a%nv)+no)-eigs((l%no))-eigs((j%no)) - ee/27.2113));

													 e10 -= 0.25*(-1.0)*real(bai(a,i)*bai(b,j)*(Vi[(d+tno)*tn3+(b+tno)*tn2+(a+tno)*tn+l]-Vi[(d+tno)*tn3+(b+tno)*tn2+l*tn+(a+tno)])*(Vi[j*tn3+(d+tno)*tn2+l*tn+i]-Vi[j*tn3+(d+tno)*tn2+i*tn+l])/(eigs((d%nv)+no)+eigs((b%nv)+no)-eigs((l%no))-eigs((i%no)) - ee/27.2113));

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
													 e4 -= 0.25*(-1.0)*real(bai(a,i)*bai(b,j)*(Vi[i*tn3+(e+tno)*tn2+j*tn+m]-Vi[i*tn3+(e+tno)*tn2+m*tn+j])*(Vi[(a+tno)*tn3+(e+tno)*tn2+(b+tno)*tn+m]-Vi[(a+tno)*tn3+(e+tno)*tn2+m*tn+(b+tno)])/(eigs((a%nv)+no)+eigs((e%nv)+no)-eigs((j%no))-eigs((m%no)) - ee/27.2113));

	 e13 -= 0.25*(-1.0)*real(bai(a,i)*bai(b,j)*(Vi[(b+tno)*tn3+(e+tno)*tn2+(a+tno)*tn+m]-Vi[(b+tno)*tn3+(e+tno)*tn2+m*tn+(a+tno)])*(Vi[j*tn3+(e+tno)*tn2+i*tn+m]-Vi[j*tn3+(e+tno)*tn2+m*tn+i])/(eigs((b%nv)+no)+eigs((e%nv)+no)-eigs((i%no))-eigs((m%no)) - ee/27.2113));

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
	 e12 -= 0.25*(-1.0)*real(bai(a,i)*bai(b,j)*(Vi[i*tn3+(b+tno)*tn2+l*tn+m]-Vi[i*tn3+(b+tno)*tn2+m*tn+l])*(Vi[j*tn3+(a+tno)*tn2+l*tn+m]-Vi[j*tn3+(a+tno)*tn2+m*tn+l])/(eigs((a%nv)+no)+eigs((b%nv)+no)-eigs((l%no))-eigs((m%no)) - ee/27.2113));

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
*/

	 cout << "Term1 - Ex State Corr:    " << dE1 << endl;
	 cout << "Term2 - OOV:              " << dE2 << endl;
	 cout << "Term3 - VVO:              " << dE3 << endl;
	 cout << "Term4 - OV:               " << dE4 << endl;
	 cout << "*Total CIS(D) correction: " << (dE1+dE2+dE3+dE4) << " au -- " << (dE1+dE2+dE3+dE4)*27.2113 << " eV" << endl;
	 cout << "*Mp2:                     " << Emp2 << endl;
	 cout << "*Total:                   " << ee+(dE1+dE2+dE3+dE4)*27.2113 << " eV" << endl;
}

cx_mat MakeRho(cx_mat Rho_, cx_mat cia, cx_double c0) // Should take C0 and Cia as arguments returns the density matrix.
{
	  Rho_.resize(n,n);
 	  Rho_.zeros();
		Rho_.eye(); Rho_.submat(no,no,n-1,n-1) *= 0.0;
		cx_mat RhoHF(Rho_);
		Rho_ *= conj(c0)*c0;

		Rho_.submat(0,no,no-1,n-1) += 1/sqrt(2)*conj(c0)*cia;
		Rho_.submat(no,0,n-1,no-1) += 1/sqrt(2)*conj(cia)*c0;

		cx_mat eRho(Rho_); eRho.zeros();
		cx_mat hRho(Rho_); hRho.zeros();


		for (int a = 0; a < nv; a++)
				for(int i = 0; i< no; i++)
						Rho_ += conj(cia(i,a))*cia(i,a)*RhoHF;

		for (int a = 0; a < nv; a++)
				for(int i = 0; i< no; i++)
						for(int ap = 0; ap < nv; ap++)
						{
								Rho_(ap+no,a+no) += 0.5*conj(cia(i,ap))*cia(i,a);
//                    eRho(a+no,ap+no) -= conj(cia(i,a))*cia(i,ap);
						}


		for (int a = 0; a < nv; a++)
				for(int i = 0; i< no; i++)
								for(int ip = 0; ip < no; ip++)
								{
										Rho_(i,ip) -= 0.5*cia(i,a)*conj(cia(ip,a));
//                        hRho(i,ip) -= cia(i,a)*conj(cia(ip,a));
								}

//        cout << "Trace Rho: " << trace(Rho) << endl;
//        cout << "Trace eRho: " << trace(eRho) << endl;
//        cout << "Trace hRho: " << trace(hRho) << endl;
//        cout << "c0^2: " << conj(c0)*c0 << endl;
//        cout << "cia^2: " << sum(conj(cia)%cia) << endl;

		return Rho_;
}

void CIS_step(cx_mat Rho, cx_mat CIA, cx C0, cx_mat &ciadot, cx &c0dot, const double tnow)
{
	  cx_mat dRho_(MakeRho(Rho,CIA,C0));
		cx_mat k1(cia), k2(cia), k3(cia), k4(cia);
		cx k01, k02, k03, k04;
		k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
		k01 = 0; k02 = 0; k03 = 0; k04 = 0;

		mat Xt = X.t();
		cx_mat Vt = V.t();
		cx_mat hmu = Vt*(Xt)*(H)*X*V;

		CISDOT(cia, c0, k1, k01, tnow, hmu);
		CISDOT(cia+k1*dt/2.0, c0+k01*dt/2.0, k2, k02, tnow+dt/2.0, hmu);
		CISDOT(cia+k2*dt/2.0, c0+k02*dt/2.0, k3, k03, tnow+dt/2.0, hmu);
		CISDOT(cia+k3*dt, c0+k03*dt, k4, k04, tnow+dt, hmu);

		cia += dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
		c0 += dt/6.0 * (k01 + 2.0*k02 + 2.0*k03 + k04);

		double sum1 = real(c0*conj(c0) + accu(cia%conj(cia)));
		cia /= sqrt(sum1);
		c0 /= sqrt(sum1);

		dRho_ = MakeRho(Rho,CIA,C0) - dRho_;
		cout << endl << tnow << endl;
		dRho_.print("dRho");
		cout << endl;

		cia.print("Cia");
		cout << c0 << endl;
		return;
}

void CISDOT(cx_mat CIA, cx C0, cx_mat &ciadot, cx &c0dot, const double tnow, cx_mat& hmu)
{

		const cx* Vi = IntCur.memptr();
		cx_mat h = hmu;
		ApplyField(h,tnow);

		// c0dot
		for (int a = 0; a < nv; a++)
				for(int i = 0; i< no; i++)
						c0dot += j * CIA(a, i) * h(i,no+a);
		// first term of ciadot
		for (int a = 0; a < nv; a++)
				for(int i = 0; i< no; i++)
						ciadot(i,a) += - j * (eigs(no + a)-eigs(i)) * CIA(i,a);
		// second term of ciadot
		for (int a = 0; a < nv; a++)
				for(int i = 0; i< no; i++)
						for(int ap = 0; ap < nv; ap++)
								for(int ip = 0; ip < no; ip++)
										ciadot(i,a) += - j * CIA(ip,ap) * (2.0 * Vi[(a+no) * n3 + ip * n2 + i * n + (ap+no)] - Vi[(a+no) * n3 + ip * n2 + (ap+no) * n + i]);


		// last term of ciadot
		for (int a = 0; a < nv; a++)
				for(int i = 0; i< no; i++)
						ciadot(i,a) += j * C0 * h(i,no+a);
		for (int a = 0; a < nv; a++)
				for(int i = 0; i< no; i++)
						for(int ap = 0; ap < nv; ap++)
								ciadot(i,a) += j /sqrt(2)* CIA(i,ap) * h(no+a,no+ap);

		for (int a = 0; a < nv; a++)
				for(int i = 0; i< no; i++)
						for(int ip = 0; ip < nv; ip++)
								ciadot(i,a) += -j /sqrt(2)* CIA(ip,a) * h(i,ip);

		return;
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
		c0 = j * 1.0;
		mat Xt = X.t();
		cx_mat Vt = V.t();
		cx_mat hmu = Vt*(Xt)*(H)*X*V;
		CISDOT(cia,c0,ccia,cc0,0,hmu);

		Hcis(0,0) = cc0;
		for (int i = 0; i < no*nv; i++)
				Hcis(0,i+1) = ccia(i);

		c0 = 0.0;
		for(int i = 0; i < no*nv; i++)
		{
				c0 = 0; cc0 = 0;
				cia.zeros();ccia.zeros();
				cia(i) = j * 1.0;

				CISDOT(cia,c0,ccia,cc0,0,hmu);
				Hcis(i+1,0) = cc0;
				for(int j = 0; j < no * nv; j++)
						Hcis(i+1,j+1) = ccia(j);
		}
		Hcis.print("Hcis:");
		vec hciseig = eig_sym(Hcis);
		hciseig.print();

		c0=0.0;
		cia.zeros();
}
