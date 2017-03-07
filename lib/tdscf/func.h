#ifndef FUNC_H
#define FUNC_H
#include <armadillo>


#ifdef __cplusplus
  extern "C"
  {
#endif

void i_dot_product(cx* AA, cx* BB, cx* CC, int a, int b, int c)
{
  //cx_mat A(AA,a,b);
  //cx_mat B(BB,b,c);
  //cx_mat C = A*B;
  cx_mat A(AA,b,a);
  cx_mat B(BB,c,b);
  cx_mat C = B*A;
  //cx_mat Ct = C.t();
  memcpy(CC,C.memptr(),2*sizeof(double)*a*c);

}

void r_dot_product(double* AA, double* BB, double* CC, int a, int b, int c)
{
  //cx_mat A(AA,a,b);
  //cx_mat B(BB,b,c);
  //cx_mat C = A*B;
  mat A(AA,b,a);
  mat B(BB,c,b);
  mat C = B*A;
  //cx_mat Ct = C.t();
  memcpy(CC,C.memptr(), sizeof(double)*a*c);

}

}
#ifdef __cplusplus

#endif

#endif
