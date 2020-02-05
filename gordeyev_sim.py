#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c
import jcoord
import move as m
import time

from mpl_toolkits.mplot3d import Axes3D

def maxwellian(v,T=1000.0):
    """
    Maxwellian distribution with distribution width specifed by temperature T. 
    """
    sigma=n.sqrt(c.k*T/c.m_e)
    return((1.0/n.sqrt(2.0*n.pi*sigma**2.0))*n.exp(-0.5*v**2.0/sigma**2.0))

def get_random_vel(T,mv=0.0):
    """
    Randomize a velocity vector of an electron with distribution standard deviation
    specified by temperature T.
    """
    sigma=n.sqrt(c.k*T/c.m_e)

    # random direction
    d=n.random.randn(3)
    d=d/n.sqrt(n.dot(d,d))

    return(n.random.randn(3)*sigma + mv*d)

def plot_maxwellian():
    """
    Plot a Maxwellian distribution of particle velocities in one direction
    """
    T=1000.0
    v=n.linspace(-1e6,1e6,num=1000)
    plt.plot(v,maxwellian(v,T=T))
    plt.show()

plot_maxwellian()

lam=c.c/430e6
lat=18.3464
lon=-66.7528
#k0=jcoord.enu2ecef(lat, lon, 0.0, 0.0, 0.0, -1.0)
k0=-jcoord.azel_ecef(lat, lon, 0, 0.0, 80)

kmag=(2.0*2.0*n.pi/lam)
k=kmag*k0
print(k0.shape)
k.shape=(3,1)
pos0=jcoord.geodetic2ecef(lat,lon,300e3)
Bxyz=m.get_B(pos0)
Bmag=n.sqrt(n.dot(Bxyz,Bxyz))
B0=Bxyz/Bmag
kpar=n.dot(k[:,0],B0)
kperp=n.sqrt(kmag**2.0-kpar**2.0)
print(kpar)
print(kperp)

n_t=200
z=n.zeros((n_t,1),dtype=n.complex64)
T=1000.0
dt=1.0/100e6
tau=n.arange(n_t)*dt
mv=n.sqrt(2*21.0*c.eV/c.m_e)
print(mv)


pos = n.zeros([100000,3])

for i in range(100000):
    t0=time.time()
    p=pos0+n.random.randn(3)*1e3

    x=m.move(x=p,B=Bxyz,nit=n_t,dt=dt,v=get_random_vel(T,mv=mv))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[:,0],x[:,1],x[:,2])
    plt.show()
    
    t1=time.time()
    print(t1-t0)
    
   # print(x.shape)
  #  print(k.shape)
    kx=n.dot(x,k)
    kx=kx-kx[0]
#    plt.plot(kx[:,0])
 #   plt.show()
    
#    plt.plot(kx)
 #   plt.show()
    z0=n.exp(1j*kx)
    z=z+z0
    print(z0.shape)
    if i%1000==0:
        plt.plot(pos[0:i,0])
        plt.show()
        C=n.sqrt(c.k*T/c.m_e)
        plt.plot(z.real/float(i+1),label="Markov")
#        plt.plot(z.imag/float(i+1))
        Omega=c.e*Bmag/c.m_e
#        plt.plot(n.exp(-0.5*kpar**2.0*C**2.0*tau**2.0)*n.exp(-(2.0*kperp**2.0*C**2.0/Omega**2.0)*n.sin(Omega*tau/2.0)**2.0))


        print("kpar %1.2f"%(kpar))
        print("kperp %1.2f"%(kperp))
#        plt.plot(n.exp(-0.5*kpar**2.0*mv**2.0*tau**2.0))

        # the gordoyev integral seems to be a sinc function convolved with a gaussian
        acft=n.exp(-0.5*kpar**2.0*C**2.0*tau**2.0)*n.sin(0.5*n.pi*kpar*mv*tau)/(0.5*n.pi*kpar*mv*tau)

        plt.plot(acft.real,label="analytic")
        plt.legend()
 #       plt.plot(acft.imag)
        plt.show()
        acft[0]=1.0
        F=n.fft.fftshift(n.fft.fft(acft))
        fvec=n.fft.fftshift(n.fft.fftfreq(n_t,d=dt))
        plt.plot(fvec/1e6,n.abs(F))
        plt.show()
        

