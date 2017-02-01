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
/*
class TDSCF{
public:
  int nn;

  TDSCF(){
    cout << "HELLO" << endl;

  }
  void testsave(int n, int n2)
  {
    nn = n + n2;
  }

  void testoutput()
  {
    cout <<"testOutput"<< nn << endl;
  }
}
*/
    // Include cpp function here as well
    int test1(int n, int n2);
    void test1_2();
    void test2();
    void test3(double *a, double *b, double *c);
    void test4(std::complex<double> *a, std::complex<double>  *b, std::complex<double>  *c,int n, int n2);


    void Update1(double *h, double *s, double *x, double *b, std::complex<double> *f, int n_, int n_aux_,int nocc_, double Enuc, double dt);
    void InitFock(cx_mat& Rho,cx_mat& V_);


}
    //}
    cx_mat FockBuild(cx_mat& Rho, cx_mat& V_);
    void Split_RK4_Step_MMUT(const arma::vec& eigval, const arma::cx_mat& Cu , const arma::cx_mat& oldrho, arma::cx_mat& newrho, const double tnow, const double dt);
    std::complex<double> RhoDot(const cx_mat& Rho_, cx_mat& RhoDot_, const double& time);
#ifdef __cplusplus

#endif



#endif
