#!/usr/bin/env python

import numpy as n
import h5py
import glob
import matplotlib.pyplot as plt

fl=glob.glob("ge*.h5")

z=n.zeros(1000,dtype=n.complex128)
for f in fl:
    h=h5py.File(f,"r")
    z=z+h["z"].value/h["i"].value
    print(len(h["z"].value/h["i"].value))
    print(h.keys())
    h.close()
z=z/float(len(fl))
plt.plot(n.abs(n.fft.fftshift(n.fft.fft(z))))
plt.show()
plt.plot(z.real)
plt.plot(z.imag)
plt.show()
