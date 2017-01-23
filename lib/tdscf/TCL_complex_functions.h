#ifndef TCL_complex_functions_h
#define TCL_complex_functions_h

#include <complex>
#include <iostream>
#include <iomanip>
#include <cmath>

#ifdef _WIN32
#define isnan(x) _isnan((x))
#define isinf(x) (!_finite(x) && !_isnan(x))
#define rint(x) (double)((int)(x))
#define NOMINMAX
#define M_LN2 0.69314718055994530942
#define M_PI_2 1.5707963267948965580
#define M_1_PI 0.31830988618379067154
#endif


using namespace std; 

inline bool FileExists(const char* name)
{
	bool found= false;
	try{
		FILE* file;
		found = (file = fopen(name, "r"));
        fclose(file);
	}
	catch(...)
	{ }
	return found;
}

static inline void loadbar(unsigned int x, unsigned int n, unsigned int w = 50)
{
    if (n==0 || x==0 || n/100 == 0)
        return;
    if ( (x != n) && (x % (n/100) != 0) ) return;
    
    float ratio  =  x/(float)n;
    int   c      =  ratio * w;
    cout << setw(3) << (int)(ratio*100) << "% [";
    for (int x=0; x<c; x++) cout << "=";
    for (int x=c; x<w; x++) cout << " ";
    cout << "]\r" << flush;
}

/*
%
%                 d
%        Psi(z) = --log(Gamma(z))
%                 dz
*/
// source: http://www.mathworks.com/matlabcentral/fileexchange/978-special-functions-math-library/content/psi.m
inline complex<double> DiGamma(const complex<double>& z_arg)
{
    complex<double> z(z_arg);
	complex<double> zz=z;
	complex<double> f(0.0,0.0);
	
	// Reflection. 
	bool refl=false; 
	if (real(z)<0.5)
	{
		z=1.0-z; 
		refl=true; 
	}
	double g=607.0/128.0; 
	static double digam_c[15] = {  0.99999999999999709182,
		57.156235665862923517,
		-59.597960355475491248,
		14.136097974741747174,
		-0.49191381609762019978,
		.33994649984811888699e-4,
		.46523628927048575665e-4,
		-.98374475304879564677e-4,
		.15808870322491248884e-3,
		-.21026444172410488319e-3,
		.21743961811521264320e-3,
		-.16431810653676389022e-3,
		.84418223983852743293e-4,
		-.26190838401581408670e-4,
		.36899182659531622704e-5};
	int order = 15;	
	complex<double> dz,dd,d,n,gg;
	for (int k=order-1; k>0;--k)  
	{	
		dz = 1./(z+complex<double>(k+1,0)-complex<double>(2,0));
		dd=digam_c[k]*dz;
		d=d+dd;
		n=n-dd*dz;
	}
	d=d+digam_c[0];
	gg=z+g-complex<double>(0.5,0.0);
	f = log(gg) + (n/d - g/gg);
	if (refl)
		f = f-M_PI*1.0/tan(M_PI*zz);
	return f;
}

inline complex<double> HarmonicNumber(const complex<double>& z_arg)
{
    complex<double> z(z_arg+1.0);
    complex<double> zz(z);
	complex<double> f(0.0,0.0);
    static complex<double> c0p5(0.5,0.0);
    static complex<double> c1(1,0.0);
    static complex<double> c2(2,0.0);
    static complex<double> c3(3,0.0);
    static complex<double> c4(4,0.0);
    static complex<double> c5(5,0.0);
    static complex<double> c6(6,0.0);
    static complex<double> c7(7,0.0);
    static complex<double> c8(8,0.0);
    static complex<double> c9(9,0.0);
    static complex<double> c10(10,0.0);
    static complex<double> c11(11,0.0);
    static complex<double> c12(12,0.0);
    static complex<double> c13(13,0.0);
    static complex<double> c14(14,0.0);
    static complex<double> c15(15,0.0);
	static double g=607.0/128.0;
	static double digam_c[15] = {  0.99999999999999709182,
		57.156235665862923517,
		-59.597960355475491248,
		14.136097974741747174,
		-0.49191381609762019978,
		.33994649984811888699e-4,
		.46523628927048575665e-4,
		-.98374475304879564677e-4,
		.15808870322491248884e-3,
		-.21026444172410488319e-3,
		.21743961811521264320e-3,
		-.16431810653676389022e-3,
		.84418223983852743293e-4,
		-.26190838401581408670e-4,
		.36899182659531622704e-5};
    // Reflection.
	bool refl=false;
	if (real(z)<0.5)
	{
		z=1.0-z;
		refl=true;
	}
	static int order = 15;
	complex<double> dz,dd,d,n,gg;
    // unroll this loop.
	{
		dz = 1./(z+c15-c2);
		dd=digam_c[14]*dz;
		d=d+dd;
		n=n-dd*dz;
        
		dz = 1./(z+c14-c2);
		dd=digam_c[13]*dz;
		d=d+dd;
		n=n-dd*dz;
        
        dz = 1./(z+c13-c2);
		dd=digam_c[12]*dz;
		d=d+dd;
		n=n-dd*dz;
        
        dz = 1./(z+c12-c2);
		dd=digam_c[11]*dz;
		d=d+dd;
		n=n-dd*dz;
        
        dz = 1./(z+c11-c2);
		dd=digam_c[10]*dz;
		d=d+dd;
		n=n-dd*dz;
        
        dz = 1./(z+c10-c2);
		dd=digam_c[9]*dz;
		d=d+dd;
		n=n-dd*dz;
        
        dz = 1./(z+c9-c2);
		dd=digam_c[8]*dz;
		d=d+dd;
		n=n-dd*dz;
        
        dz = 1./(z+c8-c2);
		dd=digam_c[7]*dz;
		d=d+dd;
		n=n-dd*dz;
        
        dz = 1./(z+c7-c2);
		dd=digam_c[6]*dz;
		d=d+dd;
		n=n-dd*dz;
        
        dz = 1./(z+c6-c2);
		dd=digam_c[5]*dz;
		d=d+dd;
		n=n-dd*dz;
        
        dz = 1./(z+c5-c2);
		dd=digam_c[4]*dz;
		d=d+dd;
		n=n-dd*dz;
        
        dz = 1./(z+c4-c2);
		dd=digam_c[3]*dz;
		d=d+dd;
		n=n-dd*dz;
        
        dz = 1./(z+c3-c2);
		dd=digam_c[2]*dz;
		d=d+dd;
		n=n-dd*dz;
        
        dz = 1./(z+c2-c2);
		dd=digam_c[1]*dz;
		d=d+dd;
		n=n-dd*dz;
    }
    d=d+digam_c[0];
	gg=z+g-c0p5;
	f = log(gg) + (n/d - g/gg);
	if (refl)
		f = f-M_PI*1.0/tan(M_PI*zz);
	return f+0.57721566490153286555;
}

inline complex<double> HarmonicNumberSlow(const complex<double>& z_arg)
{
    complex<double> z(z_arg+1.0);
    complex<double> zz(z);
	complex<double> f(0.0,0.0);
	// Reflection.
	bool refl=false;
	if (real(z)<0.5)
	{
		z=1.0-z;
		refl=true;
	}
	static double g=607.0/128.0;
	static double digam_c[15] = {  0.99999999999999709182,
		57.156235665862923517,
		-59.597960355475491248,
		14.136097974741747174,
		-0.49191381609762019978,
		.33994649984811888699e-4,
		.46523628927048575665e-4,
		-.98374475304879564677e-4,
		.15808870322491248884e-3,
		-.21026444172410488319e-3,
		.21743961811521264320e-3,
		-.16431810653676389022e-3,
		.84418223983852743293e-4,
		-.26190838401581408670e-4,
		.36899182659531622704e-5};
	int order = 15;
	complex<double> dz,dd,d,n,gg;
	for (int k=order-1; k>0;--k)
	{
		dz = 1./(z+complex<double>(k+1,0)-complex<double>(2,0));
		dd=digam_c[k]*dz;
		d=d+dd;
		n=n-dd*dz;
	}
	d=d+digam_c[0];
	gg=z+g-complex<double>(0.5,0.0);
	f = log(gg) + (n/d - g/gg);
	if (refl)
		f = f-M_PI*1.0/tan(M_PI*zz);
	return f+0.57721566490153286555;
}

inline void TestDigamma()
{
	cout << "Hi, testing digamma." << endl; 
	cout << DiGamma(complex<double>(1.1,0.0)) << " " << -0.42375494041107675258 << endl; 
	cout << DiGamma(complex<double>(1.1,1.1)) << " " << complex<double>(0.21327057874963153972, 1.04687327831605525574) << endl; 	
	
	cout << "Hi, testing HarmonicNumer: " << endl; 
	cout << HarmonicNumber(complex<double>(0.7,0.3)) << " " << complex<double>(0.812321, 0.234072) << endl; 
	cout << HarmonicNumber(complex<double>(-0.7,-0.1)) << " " << complex<double>(-2.58601, -1.11303) << endl; 		
}

// Infinite norm of a complex number.
// ----------------------------------
// It is max(|Re[z]|,|Im[z]|)

inline double inf_norm (const complex<double> &z)
{
  return max (abs (real (z)),abs (imag (z)));
}


// Test of finiteness of a complex number
// --------------------------------------
// If real or imaginary parts are finite, true is returned.
// Otherwise, false is returned

inline bool isfinite (const complex<double> &z)
{
  const double x = real(z), y = imag(z);
  return ((x==x) && (y==y));
}

// Usual operator overloads of complex numbers with integers
// ---------------------------------------------------------
// Recent complex libraries do not accept for example z+n or z==n with n integer, signed or unsigned.
// The operator overload is done here, by simply putting a cast on double to the integer.

inline complex<double> operator + (const complex<double> &z,const int n)
{
  return (z+static_cast<double> (n));
}

inline complex<double> operator - (const complex<double> &z,const int n)
{
  return (z-static_cast<double> (n));
}

inline complex<double> operator * (const complex<double> &z,const int n)
{
  return (z*static_cast<double> (n));
}

inline complex<double> operator / (const complex<double> &z,const int n)
{
  return (z/static_cast<double> (n));
}

inline complex<double> operator + (const int n,const complex<double> &z)
{
  return (static_cast<double> (n)+z);
}

inline complex<double> operator - (const int n,const complex<double> &z)
{
  return (static_cast<double> (n)-z);
}

inline complex<double> operator * (const int n,const complex<double> &z)
{
  return (static_cast<double> (n)*z);
}

inline complex<double> operator / (const int n,const complex<double> &z)
{
  return (static_cast<double> (n)/z);
}

inline complex<double> operator + (const complex<double> &z,const unsigned int n)
{
  return (z+static_cast<double> (n));
}

inline complex<double> operator - (const complex<double> &z,const unsigned int n)
{
  return (z-static_cast<double> (n));
}

inline complex<double> operator * (const complex<double> &z,const unsigned int n)
{
  return (z*static_cast<double> (n));
}

inline complex<double> operator / (const complex<double> &z,const unsigned int n)
{
  return (z/static_cast<double> (n));
}

inline complex<double> operator + (const unsigned int n,const complex<double> &z)
{
  return (static_cast<double> (n)+z);
}

inline complex<double> operator - (const unsigned int n,const complex<double> &z)
{
  return (static_cast<double> (n)-z);
}

inline complex<double> operator * (const unsigned int n,const complex<double> &z)
{
  return (static_cast<double> (n)*z);
}

inline complex<double> operator / (const unsigned int n,const complex<double> &z)
{
  return (static_cast<double> (n)/z);
}

inline bool operator == (const complex<double> &z,const int n)
{
  return (z == static_cast<double> (n));
}

inline bool operator != (const complex<double> &z,const int n)
{
  return (z != static_cast<double> (n));
}

inline bool operator == (const int n,const complex<double> &z)
{
  return (static_cast<double> (n) == z);
}

inline bool operator != (const int n,const complex<double> &z)
{
  return (static_cast<double> (n) != z);
}

inline bool operator == (const complex<double> &z,const unsigned int n)
{
  return (z == static_cast<double> (n));
}

inline bool operator != (const complex<double> &z,const unsigned int n)
{
  return (z != static_cast<double> (n));
}

inline bool operator == (const unsigned int n,const complex<double> &z)
{
  return (static_cast<double> (n) == z);
}

inline bool operator != (const unsigned int n,const complex<double> &z)
{
  return (static_cast<double> (n) != z);
}

// Logarithm of Gamma[z] && Gamma inverse function
// ------------------------------------------------
// For log[Gamma[z]], if z is not finite or is a negative integer, the program returns an error message && stops.
// The Lanczos method is used. Precision : ~ 1E-15
// The method works for Re[z] >= 0.5 .
// If Re[z] <= 0.5, one uses the formula Gamma[z].Gamma[1-z] = Pi/sin (Pi.z).
// log[sin(Pi.z)] is calculated with the Kolbig method (K.S. Kolbig, Comp. Phys. Comm., Vol. 4, p.221 (1972)) : 
// If z = x+iy && y >= 0, log[sin(Pi.z)] = log[sin(Pi.eps)] - i.Pi.n, with z = n + eps so 0 <= Re[eps] < 1 && n integer.
// If y > 110, log[sin(Pi.z)] = -i.Pi.z + log[0.5] + i.Pi/2 numerically so that no overflow can occur.
// If z = x+iy && y < 0, log[Gamma(z)] = [log[Gamma(z*)]]*, so that one can use the previous formula with z*.
//
// For Gamma inverse, Lanczos method is also used with Euler reflection formula.
// sin (Pi.z) is calculated as sin (Pi.(z-n)) to avoid inaccuracy
// with z = n + eps with n integer && |eps| as small as possible.
//
// Variables:
// ----------
// x,y: Re[z], Im[z]
// log_sqrt_2Pi,log_Pi : log[sqrt(2.Pi)], log(Pi).
// sum : Rational function in the Lanczos method
// log_Gamma_z : log[Gamma(z)] value.
// c : table containing the fifteen coefficients in the expansion used in the Lanczos method.
// eps,n : z = n + eps so 0 <= Re[eps] < 1 && n integer for Log[Gamma].
//         z=n+eps && n integer so |eps| is as small as possible for Gamma_inv.
// log_const : log[0.5] + i.Pi/2
// g : coefficient used in the Lanczos formula. It is here 607/128.
// z,z_m_0p5,z_p_g_m0p5,zm1 : argument of the Gamma function, z-0.5, z-0.5+g, z-1 

inline complex<double> log_Gamma (const complex<double> &z)
{
  if (!isfinite (z)) cout<<"z is not finite in log_Gamma."<<endl, abort ();

  const double x = real (z),y = imag (z);

  if ((z == rint (x)) && (x <= 0)) cout<<"z is negative integer in log_Gamma."<<endl, abort ();

  if (x >= 0.5)
  {
    const double log_sqrt_2Pi = 0.91893853320467274177,g = 4.7421875;
    const complex<double> z_m_0p5 = z - 0.5, z_pg_m0p5 = z_m_0p5 + g, zm1 = z - 1.0;
    const double c[15] = {0.99999999999999709182,
			  57.156235665862923517,
			  -59.597960355475491248,
			  14.136097974741747174,
			  -0.49191381609762019978,
			  0.33994649984811888699E-4,
			  0.46523628927048575665E-4,
			  -0.98374475304879564677E-4,
			  0.15808870322491248884E-3,
			  -0.21026444172410488319E-3,
			  0.21743961811521264320E-3,
			 -0.16431810653676389022E-3,
			  0.84418223983852743293E-4,
			  -0.26190838401581408670E-4,
			  0.36899182659531622704E-5};
      
    complex<double> sum = c[0];
    for (int i = 1 ; i < 15 ; i++) sum += c[i]/(zm1 + i);

    const complex<double> log_Gamma_z = log_sqrt_2Pi + log (sum) + z_m_0p5*log (z_pg_m0p5) - z_pg_m0p5;

    return log_Gamma_z;
  }
  else if (y >= 0.0)
  {
    const int n = (x < rint (x)) ? (static_cast<int> (rint (x)) - 1) : (static_cast<int> (rint (x)));
    const double log_Pi = 1.1447298858494002;
    const complex<double> log_const(-M_LN2,M_PI_2),i_Pi(0.0,M_PI);
    const complex<double> eps = z - n,log_sin_Pi_z = (y > 110) ? (-i_Pi*z + log_const) : (log (sin (M_PI*eps)) - i_Pi*n);

    const complex<double> log_Gamma_z = log_Pi - log_sin_Pi_z - log_Gamma (1.0 - z);
 
    return log_Gamma_z;
  }
  else
    return conj (log_Gamma (conj (z)));
}

inline complex<double> Gamma_inv (const complex<double> &z)
{
  if (!isfinite (z)) cout<<"z is not finite in Gamma_inv."<<endl, abort ();

  const double x = real (z);

  if (x >= 0.5)
  {
    const double log_sqrt_2Pi = 0.91893853320467274177,g = 4.7421875;
    const complex<double> z_m_0p5 = z - 0.5, z_pg_m0p5 = z_m_0p5 + g, zm1 = z - 1.0;
    const double c[15] = {0.99999999999999709182,
			  57.156235665862923517,
			  -59.597960355475491248,
			  14.136097974741747174,
			  -0.49191381609762019978,
			  0.33994649984811888699E-4,
			  0.46523628927048575665E-4,
			  -0.98374475304879564677E-4,
			  0.15808870322491248884E-3,
			  -0.21026444172410488319E-3,
			  0.21743961811521264320E-3,
			 -0.16431810653676389022E-3,
			  0.84418223983852743293E-4,
			  -0.26190838401581408670E-4,
			  0.36899182659531622704E-5};
      
    complex<double> sum = c[0];
    for (int i = 1 ; i < 15 ; i++) sum += c[i]/(zm1 + i);

    const complex<double> Gamma_inv_z = exp (z_pg_m0p5 - z_m_0p5*log (z_pg_m0p5) - log_sqrt_2Pi)/sum;

    return Gamma_inv_z;
  }
  else
  {
    const int n = static_cast<int> (rint (x));

    const complex<double> eps = z - n;

    if (n%2 == 0)
      return (sin (M_PI*eps)*M_1_PI)/Gamma_inv (1.0 - z);
    else
      return (-sin (M_PI*eps)*M_1_PI)/Gamma_inv (1.0 - z);
  }
}


#endif

