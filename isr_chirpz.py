#!/usr/bin/env python

# Code by Erhan Kudeki. 

import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot
np = numpy
plt = pyplot

from IPython.core.pylabtools import figsize, getfigs

from pylab import *
from numpy import *


def chirpz(g,n,dt,dw,wo):
    """transforms g(t) into G(w)
    g(t) is n-point array and output G(w) is (n/2)-points starting at wo
    dt and dw, sampling intervals of g(t) and G(w), and wo are 
    prescribed externally in an idependent manner 
    --- see Li, Franke, Liu [1991]"""
    g[0]=0.5*g[0] # first interval is over dt/2, and hence ...
    W = exp(-1j*dw*dt*arange(n)**2/2.) 
    S = exp(-1j*wo*dt*arange(n)) # frequency shift by wo
    x = g*W*S; y = conj(W) 
    x[n/2:] = 0.; y[n/2:] = y[0:n/2][::-1] # treat 2nd half of x and y specially
    xi = fft.fft(x); yi = fft.fft(y); G = dt*W*fft.ifft(xi*yi) #in MATLAB use ifft then fft (EK)
    return G[0:n/2]


# Ionospheric State
Ne=18.0e11 #Electron density (1/m^3)
B=15000.0e-9 #Magnetic Field (T)
fp=sqrt(Ne*80.6)

# Ion Composition
NH,N4,NO=0.001*Ne,0.001*Ne,0.998*Ne
Te,TH,T4,TO=1000,1000,1000,1000

# Physical Paramters (MKS):
me=9.1093826e-31 # Electron mass in kg
mH,m4,mO=1836.152*me,1836.152*4.*me,1836.152*16.*me # Ion mass

qe=1.60217653e-19 # C (Electron charge)
K=1.3806505e-23 # Boltzmann cobstant m^2*kg/(s^2*K);
eps0=8.854187817e-12 # F/m (Free-space permittivity)
c=299.792458e6 # m/s (Speed of light)
re=2.817940325e-15 # Electron radius

Ce,CH,C4,CO=sqrt(K*Te/me),sqrt(K*TH/mH),sqrt(K*T4/m4),sqrt(K*TO/mO) # Thermal speeds (m/s)
Omge,OmgH,Omg4,OmgO=qe*B/me,qe*B/mH,qe*B/m4,qe*B/mO # Gyro-frequencies

# Debye Lengths
debe,debH,deb4,debO=sqrt(eps0*K*Te/(Ne*qe**2)),sqrt(eps0*K*TH/(NH*qe**2)), \
        sqrt(eps0*K*T4/(N4*qe**2)),sqrt(eps0*K*TO/(NO*qe**2))
debp=1./sqrt(1./debe/debe+1./debH/debH+1./deb4/deb4+1./debO/debO) # Plasma Debye Length

# the following pseudocode is for Coulomb collision of species s with species p 
# vTsp=sqrt(2*(Cs**2+Cp**2)) #most probable interaction speed of s={e,H,O} with p={e,H,O}
# msp=ms*mp/(ms+mp) #reduced mass for s and p
# bm_sp=qs*qp/(4*pi*eps0)/msp/vTsp**2 #bmin for s and p
# log_sp=log(debp/bm_sp) #coulomb logarithm for s and p
# nusp=Np*qs**2*qp**2*log_sp/(3*pi**(3/2)*eps0**2*ms*msp*vTsp**3) # collision freq of s with p --- 2.104 Callen 2003

# electron-electron
vTee=sqrt(2.*(Ce**2+Ce**2))
mee=me*me/(me+me)
bm_ee=qe*qe/(4*pi*eps0)/mee/vTee**2
log_ee=log(debp/bm_ee)
nuee=Ne*qe**2*qe**2*log_ee/(3*pi**(3/2)*eps0**2*me*mee*vTee**3) 
# electron-hydrogen
vTeH=sqrt(2*(Ce**2+CH**2))
meH=me*mH/(me+mH)
bm_eH=qe*qe/(4*pi*eps0)/meH/vTeH**2
log_eH=log(debp/bm_eH)
nueH=NH*qe**2*qe**2*log_eH/(3*pi**(3/2)*eps0**2*me*meH*vTeH**3)
# electron-helium
vTe4=sqrt(2*(Ce**2+C4**2))
me4=me*m4/(me+m4)
bm_e4=qe*qe/(4*pi*eps0)/me4/vTe4**2
log_e4=log(debp/bm_e4)
nue4=N4*qe**2*qe**2*log_e4/(3*pi**(3/2)*eps0**2*me*me4*vTe4**3)
# electron-oxygen
vTeO=sqrt(2*(Ce**2+CO**2))
meO=me*mO/(me+mO)
bm_eO=qe*qe/(4*pi*eps0)/meO/vTeO**2
log_eO=log(debp/bm_eO)
nueO=NO*qe**2*qe**2*log_eO/(3*pi**(3/2)*eps0**2*me*meO*vTeO**3)
# electron Coulomb collision frequency
nue=nuee+nueH+nue4+nueO
nuel=nueH+nue4+nueO
nuep=nuel+nuee

# hydrogen-electron
vTHe=sqrt(2*(CH**2+Ce**2))
mHe=mH*me/(mH+me)
bm_He=qe*qe/(4*pi*eps0)/mHe/vTHe**2
log_He=log(debp/bm_He)
nuHe=Ne*qe**2*qe**2*log_He/(3*pi**(3/2)*eps0**2*mH*mHe*vTHe**3)
# hydrogen-hydrogen
vTHH=sqrt(2.*(CH**2+CH**2))
mHH=mH*mH/(mH+mH)
bm_HH=qe*qe/(4*pi*eps0)/mHH/vTHH**2
log_HH=log(debp/bm_HH)
nuHH=NH*qe**2*qe**2*log_HH/(3*pi**(3/2)*eps0**2*mH*mHH*vTHH**3) 
# hydrogen-helium
vTH4=sqrt(2*(CH**2+C4**2))
mH4=mH*m4/(mH+m4)
bm_H4=qe*qe/(4*pi*eps0)/mH4/vTH4**2
log_H4=log(debp/bm_H4)
nuH4=N4*qe**2*qe**2*log_H4/(3*pi**(3/2)*eps0**2*mH*mH4*vTH4**3)
# hydrogen-oxygen
vTHO=sqrt(2*(CH**2+CO**2))
mHO=mH*mO/(mH+mO)
bm_HO=qe*qe/(4*pi*eps0)/mHO/vTHO**2
log_HO=log(debp/bm_HO)
nuHO=NO*qe**2*qe**2*log_HO/(3*pi**(3/2)*eps0**2*mH*mHO*vTHO**3)
# hydrogen Coulomb collision frequency
nuH=nuHe+nuHH+nuH4+nuHO

# helium-electron
vT4e=sqrt(2*(C4**2+Ce**2))
m4e=m4*me/(m4+me)
bm_4e=qe*qe/(4*pi*eps0)/m4e/vT4e**2
log_4e=log(debp/bm_4e)
nu4e=Ne*qe**2*qe**2*log_4e/(3*pi**(3/2)*eps0**2*m4*m4e*vT4e**3)
# helium-hydrogen
vT4H=sqrt(2.*(C4**2+CH**2))
m4H=m4*mH/(m4+mH)
bm_4H=qe*qe/(4*pi*eps0)/m4H/vT4H**2
log_4H=log(debp/bm_4H)
nu4H=NH*qe**2*qe**2*log_4H/(3*pi**(3/2)*eps0**2*m4*m4H*vT4H**3) 
# helium-helium
vT44=sqrt(2*(C4**2+C4**2))
m44=m4*m4/(m4+m4)
bm_44=qe*qe/(4*pi*eps0)/m44/vT44**2
log_44=log(debp/bm_44)
nu44=N4*qe**2*qe**2*log_44/(3*pi**(3/2)*eps0**2*m4*m44*vT44**3)
# helium-oxygen
vT4O=sqrt(2*(C4**2+CO**2))
m4O=m4*mO/(m4+mO)
bm_4O=qe*qe/(4*pi*eps0)/m4O/vT4O**2
log_4O=log(debp/bm_4O)
nu4O=NO*qe**2*qe**2*log_4O/(3*pi**(3/2)*eps0**2*m4*m4O*vT4O**3)
# helium Coulomb collision frequency
nu4=nu4e+nu4H+nu44+nu4O

# oxygen-electron
vTOe=sqrt(2*(CO**2+Ce**2))
mOe=mO*me/(mO+me)
bm_Oe=qe*qe/(4*pi*eps0)/mOe/vTOe**2
log_Oe=log(debp/bm_Oe)
nuOe=Ne*qe**2*qe**2*log_Oe/(3*pi**(3/2)*eps0**2*mO*mOe*vTOe**3)
# oxygen-hydrogen
vTOH=sqrt(2*(CO**2+CH**2))
mOH=mO*mH/(mO+mH)
bm_OH=qe*qe/(4*pi*eps0)/mOH/vTOH**2
log_OH=log(debp/bm_OH)
nuOH=NH*qe**2*qe**2*log_OH/(3*pi**(3/2)*eps0**2*mO*mOH*vTOH**3)
# oxygen-helium
vTO4=sqrt(2*(CO**2+C4**2))
mO4=mO*m4/(mO+m4)
bm_O4=qe*qe/(4*pi*eps0)/mO4/vTO4**2
log_O4=log(debp/bm_O4)
nuO4=N4*qe**2*qe**2*log_O4/(3*pi**(3/2)*eps0**2*mO*mO4*vTO4**3)
# oxygen-osxygen
vTOO=sqrt(2.*(CO**2+CO**2))
mOO=mO*mO/(mO+mO)
bm_OO=qe*qe/(4*pi*eps0)/mOO/vTOO**2
log_OO=log(debp/bm_OO)
nuOO=NO*qe**2*qe**2*log_OO/(3*pi**(3/2)*eps0**2*mO*mOO*vTOO**3) 
# oxygen Coulomb collision frequency
nuO=nuOe+nuOH+nuO4+nuOO


Tmax=10*1.0e-6 #total integration time for electron Gordeyev integral 
N=4096*256
dt=Tmax/N

fo=0e6
fmax=12.0e6 # Hz units (I choose this)
df=(fmax-fo)/(N/2) # in Hz units - only N/2 elements are returned from chirpz
wo=2*pi*fo
dw=2*pi*df
w=wo+arange(N/2)*dw

fradar=430.0e6 # Radar Frequency (Hz)
lam=c/fradar/2.
kB=2*pi/lam # Bragg wavenumber kB = 2*ko
aspect=45.*pi/180. # Aspect angle (rad) with 0 perp to Bs
#aspect= 0.1*pi/180. # Aspect angle (rad) with 0 perp to Bs

#Electron Gordeyev integral (Brownian) 
t=arange(N)*dt 
#varel=(Ce*t)**2; varep=((2*Ce/Omge)*sin(Omge*t/2))**2 # collisionless
varel=((2.*Ce**2)/nuel**2)*(nuel*t-1+exp(-nuel*t)) # collisional
gam=arctan(nuep/Omge)
varep=((2.*Ce**2)/(nuep**2+Omge**2))*(cos(2*gam)+nuep*t-exp(-nuep*t)*cos(Omge*t-2*gam))
acfe=exp(-((kB*sin(aspect))**2)*varel/2.)*exp(-((kB*cos(aspect))**2)*varep/2.)
Ge=chirpz(acfe,N,dt,dw,wo) # Electron Gordeyev Integral
figure(1)
plot(t/1.e-6,acfe); xlabel('Time Lag (us)'); ylabel('Electron ACF')
figure(2)
plot(w/2./pi/1e6,real(Ge)); xlabel('Doppler Frequency (MHz)'); ylabel('Re[Electron Gordeyev]')
plot(w/2./pi/1e6,imag(Ge))
plt.show()


# Oxygen Gordeyev integral (Brownian)
dtO=dt*100
t=arange(N)*dtO #adjust dt such that full range of acfi is covered by range t

#varil=(CO*t)**2; varip=((2*CO/OmgO)*sin(OmgO*t/2))**2 # collisionless
varil=((2.*CO**2)/nuO**2)*(nuO*t-1+exp(-nuO*t)) # collisional
gam=arctan(nuO/OmgO)
varip=((2.*CO**2)/(nuO**2+OmgO**2))*(cos(2*gam)+nuO*t-exp(-nuO*t)*cos(OmgO*t-2*gam))
acfO=exp(-((kB*sin(aspect))**2)*varil/2.)*exp(-((kB*cos(aspect))**2)*varip/2.) 
GO=chirpz(acfO,N,dtO,dw,wo) # Ion Gordeyev Integral
figure(1)
plot(t/1.e-6,acfO); xlabel('Time Lag (ms)'); ylabel('O+ ACF')
figure(2)
plot(w/2./pi/1e6,real(GO)); xlabel('Doppler Frequency (MHz)'); ylabel('Re[O+ Gordeyev]') 
plot(w/2./pi/1e6,imag(GO))
plt.show()

# Helium Gordeyev integral (Brownian) 
dt4=dt*50
t=arange(N)*dt4 #adjust dt such that full range of acfi is covered by range t

#varil=(C4*t)**2; varip=((2*C4/Omg4)*sin(Omg4*t/2))**2 # collisionless
varil=((2.*C4**2)/nu4**2)*(nu4*t-1+exp(-nu4*t)) # collisional
gam=arctan(nu4/Omg4)
varip=((2.*C4**2)/(nu4**2+Omg4**2))*(cos(2*gam)+nu4*t-exp(-nu4*t)*cos(Omg4*t-2*gam))
acf4=exp(-((kB*sin(aspect))**2)*varil/2.)*exp(-((kB*cos(aspect))**2)*varip/2.) # page-337 in ppr-II equa-42
G4=chirpz(acf4,N,dt4,dw,wo) # Ion Gordeyev Integral
figure(1)
plot(t/1.e-6,acf4); xlabel('Time Lag (us)'); ylabel('He+ ACF')
figure(2)
plot(w/2./pi/1e6,real(G4)); xlabel('Doppler Frequency (MHz)'); ylabel('Re[He+ Gordeyev]') 
plot(w/2./pi/1e6,imag(G4))
plt.show()

# Hydrogen Gordeyev integral (Brownian) 
dtH=dt*20
t=arange(N)*dtH #adjust dt such that full range of acfi is covered by range t

#varil=(CH*t)**2; varip=((2*CH/OmgH)*sin(OmgH*t/2))**2
varil=((2.*CH**2)/nuH**2)*(nuH*t-1+exp(-nuH*t)) # page-337 in ppr-II equa-43
gam=arctan(nuH/OmgH)
varip=((2.*CH**2)/(nuH**2+OmgH**2))*(cos(2*gam)+nuH*t-exp(-nuH*t)*cos(OmgH*t-2*gam))
acfH=exp(-((kB*sin(aspect))**2)*varil/2.)*exp(-((kB*cos(aspect))**2)*varip/2.) # page-337 in ppr-II equa-42
GH=chirpz(acfH,N,dtH,dw,wo) # Ion Gordeyev Integral
figure(1)
plot(t/1.e-6,acfH); xlabel('Time Lag (us)'); ylabel('H+ ACF')
figure(2)
plot(w/2./pi/1e6,real(GH)); xlabel('Doppler Frequency (MHz)'); ylabel('Re[H+ Gordeyev]') 
plot(w/2./pi/1e6,imag(GH))
plt.show()

# Total ISR Spectrum
yO=(1-1j*w*GO)/(kB**2*debO**2) # oxygen admittance
y4=(1-1j*w*G4)/(kB**2*deb4**2) # helium admittance
yH=(1-1j*w*GH)/(kB**2*debH**2) # hydrogen admittance
ye=(1-1j*w*Ge)/(kB**2*debe**2) # electron admittance

spec=real(Ne*2*Ge)*abs((1+yH+y4+yO)/(1+ye+yH+y4+yO))**2+ \
        real(NH*2*GH+N4*2*G4+NO*2*GO)*abs((ye)/(1+ye+yH+y4+yO))**2
plot(w/2./pi/1e6,spec); 
xlabel('Doppler Frequency (MHz)'); ylabel('ISR Spectrum')
xlim(0,0.1)
plt.show()

loglog(w/2./pi/1e6,spec);
xlabel('Doppler Frequency (MHz)'); ylabel('ISR Spectrum')
plt.show()

plot(w/2./pi/1e6,log10(spec)); 
xlabel('Doppler Frequency (MHz)'); ylabel('ISR Spectrum')
#xlim(0,0.1)
plt.show()

plot(w/2./pi/1e6,spec,'.-'); 
xlabel('Doppler Frequency (MHz)'); ylabel('ISR Spectrum')
xlim(10.31,10.32)
ylim(0,1.e6)
plt.show()
