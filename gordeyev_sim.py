#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c
import jcoord
import boris_mover as m
import time
import h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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



def arecibo_gordeyev_shell(shell_mean_eV=21.0,
                           T_K=100.0,
                           max_lag=1e-5,
                           dt=1.0/100e6,
                           volume_size=1e3,
                           plot_debug=False):
    # 
    # Numerically calculate the Gordeyev integral form
    # 
    lam=c.c/430e6
    lat=18.3464
    lon=-66.7528
    
    k0=-jcoord.azel_ecef(lat, lon, 0, 0.0, 90.0)
    kmag=(4.0*n.pi/lam)
    # Bragg scattering vector
    k=kmag*k0
    k.shape=(3,1)

    # radar position
    pos_rx=jcoord.geodetic2ecef(lat,lon,0.0)
    
    # scattering position
    pos0=jcoord.geodetic2ecef(lat,lon,300e3)
    Bxyz=m.get_B(pos0)
    Bmag=n.sqrt(n.dot(Bxyz,Bxyz))
    B0=Bxyz/Bmag
    
    # B parallel component of k
    kpar=n.dot(k[:,0],B0)
    # B perpendicular component of k
    kperp=n.sqrt(kmag**2.0-kpar**2.0)

    if plot_debug:
        # Plot the k vector and the magnetic field vector, figure out the aspect angle
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # line from radar to scattering
        ax.plot( [pos_rx[0],pos0[0]],
                 [pos_rx[1],pos0[1]],
                 [pos_rx[2],pos0[2]] )

        # line from radar to scattering along magnetic field
        ax.plot( [pos0[0],pos0[0]+B0[0]*100e3],
                 [pos0[1],pos0[1]+B0[1]*100e3],
                 [pos0[2],pos0[2]+B0[2]*100e3] )

        print("aspect angle %1.2f"%(180.0*n.arccos(n.dot(k0,B0)/(n.linalg.norm(k0)*n.linalg.norm(B0)))/n.pi))
        plt.show()

    n_t = int(max_lag/dt)
    print(n_t)
    z=n.zeros(n_t,dtype=n.complex128)
    tau=n.arange(n_t)*dt

    # 0.5*m*v**2 = E
    # E = 21 eV
    # v=sqrt(2*(21 eV)/m)
    mv=n.sqrt(2*shell_mean_eV*c.eV/c.m_e)

    for i in range(10000000):
        t0=time.time()
        # random position in scattering volume
        p=pos0+n.random.randn(3)*volume_size

        # Boris mover
        x=m.move(x=p,B=Bxyz,nit=n_t,dt=dt,v=get_random_vel(T_K,mv=mv))

        if plot_debug:
            # plot the trajectory of the particle
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(x[:,0],x[:,1],x[:,2])
            plt.show()
        
        t1=time.time()

        kx=n.dot(x,k)

        kx=kx[:,0]
        kx=kx-kx[0]
        if plot_debug:
            plt.plot(kx)
            plt.show()
        
        z0=n.exp(1j*kx)

        z=z+z0

        if i%1000==0:
            print("Saving %d-%d"%(comm.rank,i))
            ho=h5py.File("ge_T%1.2f_E%1.2f_rank_%03d.h5"%(T_K,shell_mean_eV,comm.rank),"w")
            ho["z"]=z
            ho["i"]=i+1
            ho.close()
            if False:
                # Standard deviation of thermal distribution
                C=n.sqrt(c.k*T_K/c.m_e)
                # mean Gordeyev integral estimate
                plt.plot(z.real/float(i+1),label="Markov")
                plt.plot(z.imag/float(i+1))
                acf_est = z/float(i+1)
                
                # Larmour frequency
                Omega=c.e*Bmag/c.m_e
                
                print("kpar %1.2f"%(kpar))
                print("kperp %1.2f"%(kperp))
                
                # The gordoyev integral seems to be a sinc function convolved with a gaussian
                acft=n.exp(-0.5*kpar**2.0*C**2.0*tau**2.0)*n.sin(0.5*n.pi*kpar*mv*tau)/(0.5*n.pi*kpar*mv*tau)
                
                plt.plot(acft.real,label="analytic")
                plt.legend()
                
                plt.show()
                acft[0]=1.0
                F=n.fft.fftshift(n.fft.fft(acft))
                F_est=n.fft.fftshift(n.fft.fft(acf_est))            
                fvec=n.fft.fftshift(n.fft.fftfreq(n_t,d=dt))
                plt.plot(fvec/1e6,n.abs(F))
                plt.plot(fvec/1e6,n.abs(F_est))            
                plt.xlabel("Frequency (MHz)")
                plt.title("G(f)")
                plt.show()
                

#plot_maxwellian()

arecibo_gordeyev_shell()
