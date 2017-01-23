#ifndef TCLASh
#define TCLASh
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <armadillo>
#include "math.h"

using namespace arma;

// Holds all the garbage required to
// manage an active space propagation.
struct ASParameters
{
    // These are all in the lowdin basis.
    cx_mat Vs,Vs0,Csub; // (lowdinXFock change of basis for a sub-matrix.)
    cx_mat P0c,P0cx;  // Initial core density matrix (x-basis)
    cx_mat P_proj; // Projector onto 'core' initial density (square with zeros)
    cx_mat Q_proj; // Complement of 'core' initial density.
    cx_mat Ps_proj; // Projector onto 'core' initial density (rectangular)
    cx_mat Qs_proj; // Complement of 'core' initial density. (rectangular) (it's also Vs. )
    int ActiveFirst;
    int ActiveLast;
    int ne;
    double ECore; // Energy of core density.
};

#endif
