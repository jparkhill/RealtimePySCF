#include <iostream>
#include <armadillo>
using namespace arma;
#include "tdscf.h"

// shared variable (PUBLIC)
int nn; // test

// TDSCF
int n, n_aux,n_mo,n_occ;
mat RSinv;
mat H,S,X;
cx_mat F,V,C;
cx_mat rho0;
cube BpqR;
vec eigs;
double nuclear_energy,Ehf;
double dt;

std::complex<double> j(0.0,1.0);

cx_mat Rho_, RhoM12_;

int test1(int n, int n2)
{

  // Simple Addtion code
  nn = n + n2;

  return nn;
}

void test1_2()
{
  cout << "nn(CPP)" << nn << endl;
}



void test2()
{
  // Simple cout print statement
  cout << "Hello" << endl;
}

// mat x mat
void test3(double *a, double *b, double *c){

  mat A(a,4,4);
  mat B(b,4,4);
  mat C(c,4,4);

  C = A.t()*B.t();
  mat Ct = C.t();
  memcpy(c,Ct.memptr(),sizeof(double)*16);

}
// cx_mat x cx_mat
void test4(std::complex<double> *a, std::complex<double>  *b, std::complex<double>  *c,int n, int n2){

  cx_mat A(a,n,n);
  cx_mat B(b,n,n);
  cx_mat C(c,n,n);

  C = A*B;
  cx_mat Ct = C.t();
  // memcpy update needs transpose
  memcpy(c,Ct.memptr(),2*sizeof(double)*n2);

}

void Update1(double *h, double *s, double *x, double *b, std::complex<double> *f, int n_, int n_aux_,int nocc_, double Enuc, double dt_)
{
  // save the variables that does not change over the dynamics
  // make n and naux public
  // H(AO), B(LAO), V is initialized as eye.
  // make empty eigs vector
  n_mo = n = n_;
  n_aux = n_aux_;
  n_occ = nocc_;
  dt = dt_;
  cout << "NOCC:" << n_occ << endl;
  nuclear_energy = Enuc;
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
  Rho_.print("Rho");
  Rho_.submat(n_occ,n_occ,n_mo-1,n_mo-1) *= 0.0;
  Rho_.print("Rho");
  FockBuild(Rho_,V_);

  cout << "no Vi at the moment" << endl;

  InitFock(Rho_,V_);
  // checked the equivalence
  //cout << "CPP" << endl;
  //H.print("H");
  //S.print("S");
  //X.print("X");
  //BpqR.print("BpqR");
  //cout << "n: " << n << " n_aux:" << n_aux << endl;


  memcpy(f,F.memptr(),2*sizeof(double)*n*n);
}

void InitFock(cx_mat& Rho,cx_mat& V_)
{
  //setup the matrix from arrays

  cout << "Solving the RI-HF problem to ensure self-consistency." << endl;
  double err=100.0;
  int maxit = 400; int it=1;
  rho0 = V*Rho*(V.t());
  rho0.print("Inintial Rho");

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
  //Fockbuild();

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


  cx_mat Rhot=Rhol.t();


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
  Ehf = Eone + EJ + EK + nuclear_energy;

  cx_mat Vprime; eigs.zeros();
  if (!eig_sym(eigs, Vprime, F))
  {
    cout << "eig_sym failed to diagonalize fock matrix ... " << endl;
    throw 1;
  }

  eigs.print("eigs");
  V_ = V = V*Vprime;
  C = X*V;

  // UPDATE USING MEMCPY
  // F, V_, V, C, eigs are updated




  return Vprime;
}


void MMUT_step(std::complex<double> *r, std::complex<double> *rm12, double tnow)
{
  cx_mat rtmp(r,n,n);
  cx_mat rm12tmp(r,n,n);
  Rho_ = rtmp; RhoM12_ = rm12tmp;
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

  memcpy(r,Rho_.memptr(),2*sizeof(double)*n*n);
  memcpy(rm12,RhoM12_.memptr(),2*sizeof(double)*n*n);
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

std::complex<double> RhoDot(const cx_mat& Rho_, cx_mat& RhoDot_, const double& time)
{

}
