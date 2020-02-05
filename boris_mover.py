#!/usr/bin/env python
#
# Boris mover 
# (c) 2015 Juha Vierinen
# 
import numpy as n
import scipy.constants as c
import matplotlib.pyplot as plt
from pyglow.pyglow import Point
import datetime
import jcoord
from numba import jit

def get_B(pos):
    """
    Use pyglow to get IGRF magnetic field
    in ECEF coordinates
    """
    lat_lon_h=jcoord.ecef2geodetic(pos[0], pos[1], pos[2])
    pt = Point(datetime.datetime(2000, 1, 1, 1, 0), lat_lon_h[0], lat_lon_h[1], lat_lon_h[2]/1e3)
    pt.run_igrf()
    Bxyz=jcoord.enu2ecef(lat_lon_h[0], lat_lon_h[1], lat_lon_h[2], pt.Bx, pt.By, pt.Bz)
    return(Bxyz)

# boris scheme, nit time steps
@jit
def move(x=n.array([0.0,0.0,0.0]),       # Initial position
         v=n.array([1000.0,1000.0,0.0]), # Initial velocity
         E=n.array([0.0,0,0]),           # Electric field (V/m)
         B=n.array([35000e-9,0,0]),      # Magnetic field (Tesla)
         dt=1.0/25e6,                    # time step \Delta t
         nit=10000):                     # time steps
    
    xt=n.zeros([nit,3])
    tvec=n.arange(nit)*dt
    a=n.zeros(3,dtype=n.float32)
    qm = c.elementary_charge/c.m_e    # q/m
    qp = dt*qm/2.0                    # q' = dt*q/2*m 
    for i in range(nit):              # From Wikipedia (Particle in Cell)
        h = qp*B                      # h=q'*B
        hs = n.sum(h**2.0)            # |h|^2
        s = 2.0*h/(1.0+hs)            # s          
        u = v + qp*E                  # v^{n-1/2}
        up = u + n.cross((u + (n.cross(u,h))),s) # u' = u + (u + (u x h)) x s
        v = up + qp*E                 # v^{n+1/2}
        x = x + dt*v
        xt[i,:]=x                     # store position
    return(xt)


#move(x=pos,       # Initial position
#     v=n.array([1000.0,1000.0,0.0]), # Initial velocity
#     E=n.array([0.0,0,0]),           # Electric field (V/m)
#     B=n.array([35000e-9,0,0]),      # Magnetic field (Tesla)
#     dt=1.0/25e6,                    # time step \Delta t
#     nit=10000):                     # time steps

    
# tbd implement dipole magnetic field to simulate trapped particle
