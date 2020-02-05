#!/usr/bin/env python

import numpy as n
import scipy.constants as c
import matplotlib.pyplot as plt
import scipy.integrate as si
import h5py

def gordoyev_integrand(tau,k,T,m,alpha,B,nu_cc):#,phi,psin,psic_par,psic_perp):
    """
    input parameters are a list containing:
    tau - time
    k - k-vector
    alpha - magnetic field aspect angle, radians
    B - magnetic field strength
    m - mass
    nu_cc - collisions
    """
    # convert to radians
    alpha = alpha*n.pi/180.0
    
    # electron and ion thermal speeds
    C = n.sqrt(c.k*T/m)
    Omega=c.e*B/m

    # gyrofrequency normalized to k and thermal velocity
    phi=Omega/(k*C*n.sqrt(2.0))
    
    k_par = n.sin(alpha)*k
    k_perp = n.cos(alpha)*k    

    # charged-neutral collisions
    if B > 0.0:
        K1 = n.exp(-0.5*k_par**2.0*C**2.0*tau**2.0/(nu_cc**2.0))*n.exp(-(2.0*k_perp**2.0*C**2.0/(Omega**2.0))*n.sin(Omega*tau/2.0)**2.0)
    elif nu_cc > 1.0:
        K1 = n.exp(-(k**2.0*C**2.0/nu_cc)*tau)#(nu_cc*tau - 1.0 + n.exp(-nu_cc*tau)))
    else:
        K1 = n.exp(-0.5*k**2.0*C**2.0*tau**2.0)

    return(K1)

def gordoyev_integrand_magnetic(tau,k,T,m,alpha,B):
    """
    input parameters are a list containing:
    tau - time
    k - k-vector
    alpha - magnetic field aspect angle, radians
    B - magnetic field strength
    m - mass
    nu_cc - collisions
    """
    # convert to radians
    alpha = alpha*n.pi/180.0
    
    # electron and ion thermal speeds
    C = n.sqrt(c.k*T/m)
    Omega=c.e*B/m

    # parallel and perpendicular components of B
    k_par = n.sin(alpha)*k
    k_perp = n.cos(alpha)*k    

    K1 = n.exp(-0.5*k_par**2.0*C**2.0*tau**2.0)*n.exp(-(2.0*k_perp**2.0*C**2.0/(Omega**2.0))*n.sin(Omega*tau/2.0)**2.0)
    
    return(K1)



# mean k-normalized displacement of particle after t=tau,
# assuming maxwellian distribution characterized by
# temperature and mass.
def gaussian_disp(tau,k,T=2000.0,m=c.electron_mass,alpha=None,B=None,nu_cc=None):
    vel = n.sqrt(c.k*T/m)
    return(n.exp(-0.5*k**2.0*vel**2.0*tau**2.0))

#
# Simple Simpson's rule for integration
# - Use less points at larger values for more efficient approximation:
#   tau=n.linspace(0,n.sqrt(tau_inf))**2.0
# - 
def gordoyev2(om,
              n_points=1000.0,
              f=430e6,
              m=c.electron_mass,
              T=3000.0,
              B=0.0,
              alpha=0.0,
              inf=1e-3,    # time at infinity
              displacement_func=gordoyev_integrand_magnetic):
    # monostatic k
    k = -2.0*2.0*n.pi*f/c.c
    
    # more points in the beginning
    tau = n.linspace(0,n.sqrt(inf),num=int(n_points))**2.0
    
    # tau = n.linspace(0,inf,num=int(n_points))

    # displacement of particles of some species depends on their velocity distribution.
    df=displacement_func(tau,k=k,T=T,m=m,alpha=alpha,B=B)
    val=n.exp(-1j*om*tau)*df
#    plt.plot(df.real)
#    plt.plot(df.imag)
#    plt.show()
    sint = si.simps(val,tau)
    return(sint)

# evaluate gordoyev integral on a number of frequencies
def gordoyev(om,n_points=1e4,f=430e6,m=c.electron_mass,T=3000.0,B=0.0,alpha=0.0,inf=1e-3):
    res=n.zeros(len(om),dtype=n.complex128)
    for omi,o in enumerate(om):
        print(omi)
        res[omi] = gordoyev2(o,n_points=n_points,f=f,m=m,T=T,alpha=alpha,B=B,inf=inf)
    return(res)

def debye_length(t_s,n_s):
    return(n.sqrt(c.epsilon_0*c.k*t_s/(c.e**2.0*n_s)))

def test_fractions(ion_fractions):
    n_i_sum=0.0
    for i in range(len(ion_fractions)):
        n_i_sum+=ion_fractions[i]
    if n.abs(n_i_sum-1.0) > 1e-3:
        print("ion fractions need to add up to 1!")
        exit(0)

def coulomb_collision_freq(n_e,t_e):
    ## from amisrpy
    nuei = 1e-6*54.5*n_e/t_e**1.5 # Schunk and Nagy 4.144
    nuee = 1e-6*54.5/n.sqrt(2.0)*n_e/t_e**1.5 # Schunk and Nagy 4.145
    return(nuei,nuee)
    
def isr_spectrum(om,plpar=None,n_points=1e4,ni_points=1e4):
    n_e=plpar["n_e"] # 1/m^3
    f=plpar["freq"] # Hz
    t_i=plpar["t_i"] # K
    t_e=plpar["t_e"] # K
    m_i=plpar["m_i"] # ion masses (amu)
    ion_fractions=plpar["ion_fractions"] # ion fractions (has to sum up to 1.0)
    alpha=plpar["alpha"] # aspect angle (0 = perp)
    B=plpar["B"] # magnetic field nT
#    nu_cc=plpar["nu_cc"] # collisions
    
    h_e = debye_length(t_e,n_e)
    n_ions = len(m_i)
    n_freqs = len(om)
    
    test_fractions(ion_fractions)
    
    # monostatic k
    k = -2.0*2.0*n.pi*f/c.c
    
    nu_ei,nu_ee = coulomb_collision_freq(n_e,t_e)
    print("nu_ei ",nu_ei)
    print("nu_ee ",nu_ee)    
    J_e = gordoyev(om,f=f,T=t_e,m=c.electron_mass,n_points=n_points,alpha=alpha,B=B,inf=1e-4)
    n_te2 = n_e*2.0*n.real(J_e)    
    J_i = []
    h_i = []    
    sigma_i = []
    n_ti2 = []
    n_ti2 = n.zeros(n_freqs,dtype=n.complex128)
    sigma_i = n.zeros(n_freqs,dtype=n.complex128)
    for i in range(n_ions):
        h_ion=debye_length(t_i[i],n_e*ion_fractions[i])
        J_ion = gordoyev(om,f=f,T=t_i[i],m=c.atomic_mass*m_i[i],alpha=alpha,B=B,n_points=ni_points)
        n_ti2+=(n_e*ion_fractions[i])*2.0*n.real(J_ion)
        sigma_i += 1.0j*om*c.epsilon_0 * ((1.0 - 1.0j*om*J_ion)/(h_ion**2.0*k**2.0))

    sigma_e = 1.0j*om*c.epsilon_0 * ((1.0 - 1.0j*om*J_e)/(h_e**2.0*k**2.0))
    
    il = ((n.abs(1.0j*om*c.epsilon_0 + sigma_i)**2.0)*n_te2)/(n.abs(1.0j*om*c.epsilon_0 + sigma_e + sigma_i)**2.0)
    pl = ((n.abs(sigma_e)**2.0)*n_ti2)/(n.abs(1.0j*om*c.epsilon_0 + sigma_e + sigma_i)**2.0)

    return(pl + il)
 
def pf(ne):
    return(n.sqrt(ne*c.e**2.0/(c.electron_mass*c.epsilon_0))/2.0/n.pi)

def prf(ne,f,t_e):
    k = 2.0*2.0*n.pi*f/c.c
    return(n.sqrt(pf(ne)**2.0 + 3.0*k**2.0*c.k*t_e/(c.electron_mass*4.0*n.pi**2.0)))    

def freq2ne(pf,f,t_e):
    k = 2.0*2.0*n.pi*f/c.c
    ne=((c.electron_mass*c.epsilon_0)*(2.0*n.pi)**2.0*(pf**2.0-3.0*k**2.0*c.k*t_e/(c.electron_mass*4.0*n.pi**2.0)))/c.e**2.0
    return(ne)

def freq2neb(pf,f,t_e,alpha=45.0,B=35000e-9):
    Omega=c.e*B/c.m_e/2.0/n.pi
    print(Omega/2.0/n.pi)
    alpha=n.pi*alpha/180.0
    k = 2.0*2.0*n.pi*f/c.c    
#    pf**2.0 - Omega**2.0*n.sin(alpha)**2.0 - 3.0*k**2.0*c.k*t_e/(4.0*n.pi**2.0*c.m_e) = ne*c.e**2.0/(c.electron_mass*c.epsilon_0)/(2.0**2.0*n.pi**2.0)
    a=pf**2.0 - Omega**2.0*n.sin(alpha)**2.0 - 3.0*k**2.0*c.k*t_e/(4.0*n.pi**2.0*c.m_e)
#    if a < 0:
 #       print("too low frequency, negative n_e")
    return(a*(4.0*n.pi**2.0*c.m_e*c.epsilon_0)/(c.e**2.0))


def spec_test():
    plpar={"t_i":[1000.0,1000.0],
           "t_e":1500.0,
           "m_i":[1.0,16.0],
           "n_e":2e10,
           "freq":430e6,
           "B":25000e-9,
           "alpha":45.0,
           "ion_fractions":[0.8,0.2]}

    ne=plpar["n_e"]
    pfreq=pf(ne)*2.0*n.pi
    print(pf(ne))
    print(prf(ne,430e6,1500.0))
    pl_freq = 2.0*n.pi*prf(ne,430e6,1500.0)
    om0 = 0.0
    om = n.linspace(om0-2e6*2.0*n.pi,om0+2e6*2.0*n.pi,num=1000.0)
    spec=isr_spectrum(om,plpar=plpar,n_points=1e4)
    print(spec)
    spec=(spec/n.nanmax(spec))*230.0

#    f1 = om/2.0/n.pi
 #   f_idx = n.where(n.abs(f1) < 150e3)[0]
  #  spec[f_idx]=0.0
   # print(n.sum(spec.real))  
    
    plt.plot(om/2.0/n.pi,10.0*n.log10(spec))
    plt.ylim([-15,25])

    
    plt.show()

# try to see if 
def gl_test():
    plpar={"t_i":[200.0,200.0],
           "t_e":200.0,
           "m_i":[16.0,32.0],
           "n_e":2e10,
           "freq":430e6,
           "B":35000e-9,
           "alpha":45.0,
           "ion_fractions":[0.2,0.8]}

    ne=plpar["n_e"]
    pfreq=pf(ne)*2.0*n.pi
    print(pf(ne))
    print(prf(ne,430e6,1500.0))
    pl_freq = 2.0*n.pi*prf(ne,430e6,1500.0)
    om0 = 0.0
    om = n.linspace(om0-2e6*2.0*n.pi,om0+2e6*2.0*n.pi,num=1000.0)
    spec=isr_spectrum(om,plpar=plpar,n_points=1e4)
    print(spec)
    spec=(spec/n.nanmax(spec))*230.0

#    f1 = om/2.0/n.pi
 #   f_idx = n.where(n.abs(f1) < 150e3)[0]
  #  spec[f_idx]=0.0
   # print(n.sum(spec.real))  
    
    plt.plot(om/2.0/n.pi,10.0*n.log10(spec))
    plt.ylim([-15,25])

    
    plt.show()
    
# test to see what type of a grid is needed to resolve the plasma-line
def pl_test():
    ne=3e11
    pfreq=prf(ne,430e6,1500.0)

    plpar={"t_i":[1000.0,1000.0],
           "t_e":1500.0,
           "m_i":[1.0,16.0],
           "n_e":ne,
           "freq":430e6,
           "B":25000e-9,
           "alpha":45.0,
           "ion_fractions":[0.1,0.9]}
    
    om = n.linspace(-50e3*2.0*n.pi,50e3*2.0*n.pi,num=100)
    il_spec=isr_spectrum(om,plpar=plpar,n_points=1e4)
    
    s_il=si.simps(il_spec,om)
#    plt.plot(om/2.0/n.pi,il_spec)
#    plt.show()
    
    pl_freq = 2.0*n.pi*prf(ne,430e6,1500.0)
    om0 = 0.0
    n_points=5e5
    df=0.1e3
    om = n.linspace(om0-pl_freq-200e3*2.0*n.pi,om0-pl_freq+200e3*2.0*n.pi,num=2000)
    pl_spec=isr_spectrum(om,plpar=plpar,n_points=n_points)
    plt.plot((om-om0)/2.0/n.pi/1e3,pl_spec)
    plt.show()
    om0 = om[n.argmax(pl_spec)]
    
    om = n.linspace(om0-df*2.0*n.pi,om0+df*2.0*n.pi,num=200)
    pl_spec=isr_spectrum(om,plpar=plpar,n_points=n_points)
    s0=si.simps(pl_spec,om)
    plt.plot((om-om0)/2.0/n.pi/1e3,10.0*n.log10(pl_spec),label="200")        
    
    om = n.linspace(om0-df*2.0*n.pi,om0+df*2.0*n.pi,num=400)
    pl_spec=isr_spectrum(om,plpar=plpar,n_points=n_points)
    s1=si.simps(pl_spec,om)
    plt.plot((om-om0)/2.0/n.pi/1e3,10.0*n.log10(pl_spec),label="400")        
    
    om = n.linspace(om0-df*2.0*n.pi,om0+df*2.0*n.pi,num=800)
    pl_spec=isr_spectrum(om,plpar=plpar,n_points=n_points)
    s2=si.simps(pl_spec,om)
    plt.plot((om-om0)/2.0/n.pi/1e3,10.0*n.log10(pl_spec),label="800")    
    
    om = n.linspace(om0-df*2.0*n.pi,om0+df*2.0*n.pi,num=2*1600)
    pl_spec=isr_spectrum(om,plpar=plpar,n_points=n_points)
    s3=si.simps(pl_spec,om)    
    
    plt.plot((om-om0)/2.0/n.pi/1e3,10.0*n.log10(pl_spec),label="3200")
    plt.legend()
    plt.title(om0/2.0/n.pi/1e6)
    plt.xlabel("Frequency around plasma-line (kHz)")
    # integrated ion-line to plasma-line ratio, checks for convergence
    # pl is very sharp without collisions
    print(s_il/s0)    
    print(s_il/s1)
    print(s_il/s2)
    print(s_il/s3)        
    plt.show()

# test to see what type of a grid is needed to resolve the plasma-line
def pl_test2():
    ne=1.2e10
    pfreq=pf(ne)
    print(pf(ne))
    print(prf(ne,430e6,1500.0))

    plpar={"t_i":[1000.0,1000.0],
           "t_e":1500.0,
           "m_i":[1.0,16.0],
           "n_e":ne,
           "freq":430e6,
           "B":25000e-9,
           "alpha":45.0,
           "ion_fractions":[0.9,0.1]}

    om = n.linspace(-1.2*20e3*2.0*n.pi,1.2*20e3*2.0*n.pi,num=500)
    il_spec=isr_spectrum(om,plpar=plpar,n_points=1e4)
    sf=100.0/n.max(il_spec)    
    plt.plot(om/2.0/n.pi,sf*il_spec)

    om = n.linspace(-1.6*pfreq*2.0*n.pi,1.6*pfreq*2.0*n.pi,num=4000)
    il_spec=isr_spectrum(om,plpar=plpar,n_points=1e4)   
    print(pfreq)
    
    plt.plot(om/2.0/n.pi,sf*il_spec)
    plt.show()
#    s_il=si.simps(il_spec,om)
 #   plt.plot(om/2.0/n.pi,il_spec)
  #  


  # test to see what type of a grid is needed to resolve the plasma-line
def il_test():
    ne=3e11

    plpar={"t_i":[2000.0],
           "t_e":2000.0,
           "m_i":[16.0],
           "n_e":ne,
           "freq":500e6,
           "B":25000e-9,
           "alpha":90.0,
           "ion_fractions":[1.0]}

    om = n.linspace(-1.2*20e3*2.0*n.pi,1.2*20e3*2.0*n.pi,num=500)
    il_spec=isr_spectrum(om,plpar=plpar,n_points=1e4)
    h=h5py.File("ilspec.h5","w")
    h["spec"]=il_spec
    h["om"]=om
    h.close()
    sf=100.0/n.max(il_spec)    
    plt.plot(om/2.0/n.pi/1e3,sf*il_spec)
    plt.xlabel("Doppler shift (kHz)")
    plt.ylabel("Power spectral density")
    plt.title("$m_i=%d$ (amu), $T_e=T_i=%1.0f$ K, $f=%1.0f$ MHz"%(plpar["m_i"][0],plpar["t_i"][0],plpar["freq"]/1e6))
    plt.savefig("ilex.png")
    plt.show()

def il_test2():
    ne=3e11

    plpar={"t_i":[200.0],
           "t_e":200.0,
           "m_i":[64.0],
           "n_e":ne,
           "freq":500e6,
           "B":25000e-9,
           "alpha":90.0,
           "ion_fractions":[1.0]}

    om = n.linspace(-1.2*20e3*2.0*n.pi,1.2*20e3*2.0*n.pi,num=500)
    il_spec=isr_spectrum(om,plpar=plpar,n_points=1e4)
    sf=100.0/n.max(il_spec)    
    plt.plot(om/2.0/n.pi/1e3,sf*il_spec)
    plt.xlabel("Doppler shift (kHz)")
    plt.ylabel("Power spectral density")
    plt.title("$m_i=%d$ (amu), $T_e=T_i=%1.0f$ K, $f=%1.0f$ MHz"%(plpar["m_i"][0],plpar["t_i"][0],plpar["freq"]/1e6))
    plt.savefig("ilex2.png")
    plt.show()


def il_amb():
    ne=3e11
    om = n.linspace(-1.2*20e3*2.0*n.pi,1.2*20e3*2.0*n.pi,num=500)
    plpar={"t_i":[200.0],
           "t_e":200.0,
           "m_i":[16.0],
           "n_e":ne,
           "freq":500e6,
           "B":25000e-9,
           "alpha":90.0,
           "ion_fractions":[1.0]}
    il_spec_16=isr_spectrum(om,plpar=plpar,n_points=1e4)
    plt.figure(figsize=(14,8))
    plt.subplot(121)
    plt.plot(om/2.0/n.pi/1e3,il_spec_16)
    plt.xlabel("Doppler shift (kHz)")
    plt.ylabel("Power spectral density")
    plt.title("$m_i=%d$ (amu), $T_e=T_i=%1.0f$ K, $f=%1.0f$ MHz"%(plpar["m_i"][0],plpar["t_i"][0],plpar["freq"]/1e6))    
    
    plpar={"t_i":[400.0],
           "t_e":400.0,
           "m_i":[32.0],
           "n_e":ne,
           "freq":500e6,
           "B":25000e-9,
           "alpha":90.0,
           "ion_fractions":[1.0]}
    il_spec_32=isr_spectrum(om,plpar=plpar,n_points=1e4)
    plt.subplot(122)    
    plt.plot(om/2.0/n.pi/1e3,il_spec_32)    
    plt.xlabel("Doppler shift (kHz)")
    plt.ylabel("Power spectral density")
    plt.title("$m_i=%d$ (amu), $T_e=T_i=%1.0f$ K, $f=%1.0f$ MHz"%(plpar["m_i"][0],plpar["t_i"][0],plpar["freq"]/1e6))
    plt.tight_layout()
    plt.savefig("il_amb.png")
    plt.show()
    
  
if __name__ == "__main__":
    pl_test()        
    pl_test2()    
    il_amb()
    
    il_test()
    il_test2()                

    gl_test()            
    pl_test()        
    spec_test()
    
    pl_test()    
    exit(0)
    

