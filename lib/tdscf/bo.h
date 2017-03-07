#ifndef BO_H
#define BO_H

//#include <xc.h>
//#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))


#ifdef __cplusplus
  extern "C"
  {
#endif

int nA, n_aux0;
mat U;
cube B1, B0;
mat S0;
void SetupBO(int nA_, int n_, int n_occ_, int n_aux_, int n_aux0_, double *U_ptr, double *B0_, double *B1_)
{
  #pragma omp parallel
        #pragma omp master
        {
          cout <<"OMP Threads:" << omp_get_num_threads() << endl;
        }
  nA = nA_; n = n_; n_occ = n_occ_; n_aux = n_aux_; n_aux0 = n_aux0_;
  mat U_(U_ptr,n,n);
  U = U_.t();

  cube BpqR0tmp(B0_,n_aux0,nA,nA);
  B0.resize(nA,nA,n_aux0);
  for(int i = 0; i < nA; i++)
    for(int j = 0; j < nA; j++)
      for(int k = 0; k < n_aux0; k++)
        B0(i,j,k) = BpqR0tmp(k,i,j);
  cube BpqR1tmp(B1_,n_aux,n,n);
  B1.resize(n,n,n_aux);
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
      for(int k = 0; k < n_aux; k++)
        B1(i,j,k) = BpqR1tmp(k,i,j);


}

void get_j1(std::complex<double> * P,std::complex<double> * J_)
{
  cx_mat rho(P,n,n); // AO
  mat Jtemp(n,n); Jtemp.zeros();
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))

#pragma omp declare reduction( + : mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< double >( )) ) initializer (omp_priv(omp_orig))

#pragma omp parallel for reduction(+:Jtemp) schedule(static)
  for (int R=0; R<n_aux; ++R)
  {
    mat Btemp=B1.slice(R);
    mat I1=real(rho%Btemp);
    Jtemp += Btemp*accu(I1);
  }

  memcpy(J_,Jtemp.memptr(),sizeof(double)*n*n);
}



void get_k0(std::complex<double> * P,std::complex<double> * K_)
{
  cx_mat rho(P,nA,nA); // AO
  cx_mat Ktemp(nA,nA); Ktemp.zeros();
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))

#pragma omp declare reduction( + : mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< double >( )) ) initializer (omp_priv(omp_orig))

#pragma omp parallel for reduction(+:Ktemp) schedule(static)
  for (int R=0; R<n_aux0; ++R)
  {
    mat Btemp=B0.slice(R);
    //mat I1=real(Rhol%B);
    //J += 2.0*B*accu(I1);
    Ktemp += Btemp*(rho)*Btemp;
  }

  memcpy(K_,Ktemp.memptr(),2*sizeof(double)*nA*nA);
}

}
#ifdef __cplusplus

#endif

#endif
