#ifndef TCLpolematricesh
#define TCLpolematricesh
#include <armadillo>
#include "one-e.h"

using namespace arma;

double ND(double x, double y, double z);
void NDMatrix(double* jMtrx_x,INTEGER dir, INTEGER jobType, INTEGER grdTyp);
void FieldMatricesViaNumericalntegration(double* jMtrx_x,INTEGER dir, INTEGER jobType, INTEGER grdTyp);

class FieldMatrices{
public:
	arma::cx_mat mux,muy,muz; // AO basis.
	arma::cx_mat muxo,muyo,muzo; // MO basis.
	arma::vec mu_0,pol;
    
	FieldMatrices(const arma::cx_mat& mux_,const arma::cx_mat& muy_,const arma::cx_mat& muz_,const vec& pol_): mux(mux_), muy(muy_), muz(muz_), mu_0(3), pol(pol_)
	{
		mu_0.zeros();
		return;
	}
    
    virtual ~FieldMatrices(){}
    
    virtual void Sync(const FieldMatrices* other)
    {
        mux=(other->mux);
        muy=(other->muy);
        muz=(other->muz);
        muxo=(other->muxo);
        muyo=(other->muyo);
        muzo=(other->muzo);
        mu_0=(other->mu_0);
    }
    
    void save(std::string nm)
    {
        muxo.save(nm+"muxo",arma_binary);
        muyo.save(nm+"muyo",arma_binary);
        muzo.save(nm+"muzo",arma_binary);
        mux.save(nm+"mux",arma_binary);
        muy.save(nm+"muy",arma_binary);
        muz.save(nm+"muz",arma_binary);
        mu_0.save(nm+"mu_0",arma_binary);
    }
    
    void load(std::string nm)
    {
        muxo.load(nm+"muxo");
        muyo.load(nm+"muyo");
        muzo.load(nm+"muzo");
        mux.load(nm+"mux");
        muy.load(nm+"muy");
        muz.load(nm+"muz");
        mu_0.load(nm+"mu_0");
    }
    
	void update(const arma::cx_mat& C)
	{
		muxo=C.t()*mux*C;
		muyo=C.t()*muy*C;
		muzo=C.t()*muz*C;
		muxo = 0.5*(muxo+muxo.t());
		muyo = 0.5*(muyo+muyo.t());
		muzo = 0.5*(muzo+muzo.t());
	}
    
	virtual void ApplyField(cx_mat& arg, const double& time, bool& IsOn) const = 0;
    
    virtual double ImpulseAmp(const double time) const = 0;
    
	void InitializeExpectation(const arma::cx_mat& pin)
	{
		mu_0=Expectation(pin,false);
        mu_0.print("Initial Dipole:");
	}
	
	vec Expectation(const arma::cx_mat& pin, bool Initialize=true) const
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
};

class StaticField : public FieldMatrices{
public:
	double Amp, tau, t0;
    
	StaticField(const arma::cx_mat& mux_, const arma::cx_mat& muy_, const arma::cx_mat& muz_, const vec pol_, double Amp_, double Tau_, double t0_): FieldMatrices(mux_,muy_,muz_,pol_), Amp(Amp_), tau(Tau_), t0(t0_)
	{
		mu_0.zeros();
		return;
	}
    
	void ApplyField(cx_mat& arg, const double& time, bool& IsOn) const
	{
        IsOn = (std::abs(ImpulseAmp(time)) > pow(10.0,-10.0));
		if (IsOn)
			arg += (pol(0)*muxo+pol(1)*muyo+pol(2)*muzo)*ImpulseAmp(time);
		else
			return;
	
    }
    
    double ImpulseAmp(const double time) const
	{
        if (tau!=0.0)
            return Amp*exp(-1.0*pow(time-t0,2.0)/(2.0*tau*tau));
        else
            return Amp;
	}
    
};

class OpticalField: public FieldMatrices{
public:
	bool ApplyImpulse,ApplyImpulse2;
	bool ApplyCw;
	double Amp,Freq,Amp2,Freq2;
    double tau, tau2;
    double t0, t02;

	OpticalField(const arma::cx_mat& mux_,const  arma::cx_mat& muy_, const arma::cx_mat& muz_,const vec pol_, bool ApplyImpulse_, bool ApplyCw_, double Amp_, double Freq_, double Tau_, double t0_,
                   bool ApplyImpulse2_, double Amp2_, double Freq2_, double Tau2_, double t02_): FieldMatrices(mux_,muy_,muz_,pol_),
        ApplyCw(ApplyCw_), ApplyImpulse(ApplyImpulse_), Amp(Amp_), Freq(Freq_), tau(Tau_), t0(t0_), ApplyImpulse2(ApplyImpulse2_), Amp2(Amp2_), Freq2(Freq2_), tau2(Tau2_), t02(t02_)
	{
		mu_0.zeros();
		return;
	}
 
	void ApplyField(cx_mat& arg, const double& time, bool& IsOn) const
	{
		if (ApplyCw)
			CwLaser(arg,time,IsOn);
		if (ApplyImpulse || ApplyImpulse2)
			Impulse(arg,time,IsOn);
	}
	
	// For the purposes of initializing mu no laser if t=0
	void CwLaser(cx_mat& arg, const double& time, bool& IsOn) const
	{
		if (time>0.0)
		{
			IsOn = true;
			arg += (pol(0)*muxo+pol(1)*muyo+pol(2)*muzo)*Amp*cos(Freq*time);
		}
	}
    
    double ImpulseAmp(const double time) const
	{
		if (ApplyImpulse2)
			return (Amp*sin(Freq*time)*(1.0/sqrt(2.0*M_PI*tau*tau))*exp(-1.0*pow(time-t0,2.0)/(2.0*tau*tau))+
					Amp2*sin(Freq2*time)*(1.0/sqrt(2.0*M_PI*tau2*tau2))*exp(-1.0*pow(time-t02,2.0)/(2.0*tau2*tau2)));
		else
			return Amp*sin(Freq*time)*(1.0/sqrt(2.0*M_PI*tau*tau))*exp(-1.0*pow(time-t0,2.0)/(2.0*tau*tau));
	}
    
	// Arguements are in au.
	void Impulse(cx_mat& arg, const double time,bool& IsOn) const
	{
		if (!(ApplyImpulse || ApplyImpulse2))
			return;
        
        IsOn = (std::abs(ImpulseAmp(time)) > pow(10.0,-10.0));
        
		if (IsOn)
		{
			if (pol.n_elem <3 || muxo.n_elem+muyo.n_elem+muzo.n_elem != 3*arg.n_elem)
			{
				cout << "Dipole size error..." << endl;
				throw;
			}
			arg += pol(0)*muxo*ImpulseAmp(time);
			arg += pol(1)*muyo*ImpulseAmp(time);
			arg += pol(2)*muzo*ImpulseAmp(time);
		}
		else
			return;
	}

	vec Expectation(const arma::cx_mat& pin, bool Initialize=true) const
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
};


#endif
