#ifndef FIELD_H
#define FIELD_H
#include <armadillo>


#ifdef __cplusplus
  extern "C"
  {
#endif
  cx_mat mux,muy,muz; // AO basis
  cx_mat muxo,muyo,muzo; // C's basis.
  vec mu_0,pol(3),mu_n(3);
  double Amp,Freq;
  double tau;
  double t0;
  bool ApplyCw, ApplyImpulse;


  void updatefield(cx_mat& C)
  {
    muxo=C.t()*mux*C;
		muyo=C.t()*muy*C;
		muzo=C.t()*muz*C;
		muxo = 0.5*(muxo+muxo.t());
		muyo = 0.5*(muyo+muyo.t());
		muzo = 0.5*(muzo+muzo.t());

  }
  vec Expectation(const arma::cx_mat& pin, bool Initialize=true)
	{
		vec tore(3);
		tore[0] = trace(real(pin*muxo));
		tore[1] = trace(real(pin*muyo));
		tore[2] = trace(real(pin*muzo));
        if (Initialize)
            return tore-mu_0;
        else
            return tore;
  }

  void InitializeExpectation(const arma::cx_mat& pin)
	{
		mu_0=Expectation(pin,false);
    mu_0.print("Initial Dipole (x, y, z):");
	}

  double ImpulseAmp(double tnow)
  {
    //cout << "tnow"<< tnow << endl;
    //cout << "SIN" << sin(Freq*tnow) << endl << endl;
  	return Amp*sin(Freq*tnow)*(1.0/sqrt(2.0*M_PI*tau*tau))*exp(-1.0*pow(tnow-t0,2.0)/(2.0*tau*tau));
  }

  void Impulse(cx_mat& arg, double tnow)
  {
    bool IsOn = (std::abs(ImpulseAmp(tnow)) > pow(10.0,-10.0));

		if (IsOn)
		{
			if (pol.n_elem <3 || muxo.n_elem+muyo.n_elem+muzo.n_elem != 3*arg.n_elem)
			{
				cout << "Dipole size error..." << endl;
				throw;
			}
      //cout << ImpulseAmp(tnow) << endl;
			arg += (pol(0)*muxo+pol(1)*muyo+pol(2)*muzo)*ImpulseAmp(tnow);
			//arg += pol(1)*muyo*ImpulseAmp(tnow);
			//arg += pol(2)*muzo*ImpulseAmp(tnow);
		}
		else
			return;
  }

  void CwLaser(cx_mat& arg, const double& tnow)
  {
  	if (tnow>0.0)
  	{
  		arg += (pol(0)*muxo+pol(1)*muyo+pol(2)*muzo)*Amp*cos(Freq*tnow);
  	}
  }

  void ApplyField(cx_mat& arg, double tnow)
  {
    if(ApplyImpulse)
      Impulse(arg,tnow);
    if(ApplyCw)
      CwLaser(arg,tnow);
  }

}


#ifdef __cplusplus

#endif



#endif
