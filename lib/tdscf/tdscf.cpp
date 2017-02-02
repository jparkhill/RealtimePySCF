#include <iostream>
#include <armadillo>
using namespace arma;
#include "tdscf.h"
#include "field.h"

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

  // Why am I doing this?
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
  Rho_ = 0.5*(NewRho+NewRho.t());//0.5*(NewRho+NewRho.t());
  RhoM12_ = 0.5*(NewRhoM12+NewRhoM12.t());//0.5*(NewRhoM12+NewRhoM12.t());

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


cx RhoDot(const cx_mat& Rho_, cx_mat& RhoDot_, const double& time)
{

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
