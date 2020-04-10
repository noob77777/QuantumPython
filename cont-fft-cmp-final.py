import numpy as np
import matplotlib.pyplot as plt
import math

import matplotlib.animation as animation
from matplotlib import style
style.use('ggplot')

e = math.e
PI = math.pi

def exp(x):
    return e**x

dx = 0.1
x = np.arange(-10, 10, dx).astype(complex)
y = np.sin(x).astype(complex)+np.sin(5*x).astype(complex)+np.sin(25*x).astype(complex)

dk = 2 * np.pi / (len(x) * dx)
k0 = -0.5 * len(x) * dk
k = k0 + dk * np.arange(len(x)).astype(complex)


def FT(k, x, y):
    F = np.array(x).astype(complex)
    for kk in range(len(k)):
        s = 0
        for xx in range(len(x)):
            s += y[xx]*exp(-1j*k[kk]*x[xx])*dx
        s /= (2*PI)**0.5
        F[kk] = s
    return F

def fft(k, x, y):
    F = np.array(y).astype(complex)
    F *= dx/(2*PI)**0.5
    phase = np.exp(-1j*x*k[0])
    F = F*phase
    F = np.fft.fft(F)
    F = F*np.exp(1j*x[0]*dk*np.array(range(len(k))))
    return F

def ifft(k, x, F):
    y = np.array(F)
    y = y*np.exp(-1j*x[0]*dk*np.array(range(len(k))))
    y = np.fft.ifft(y)
    phase = np.exp(1j*x*k[0])
    y = y*phase
    y *= (2*PI)**0.5/dx
    return y

        
F = FT(k, x, y)
F2 = fft(k, x, y)

z = ifft(k, x, F2)

fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

ax1.scatter([0, 0], [-10, 10], s = 1)
ax1.plot(x.real, y.real)
ax2.plot(k.real, F2.imag)
ax2.scatter([0, 0], [-10, 10], s = 1)

ax3.plot(k.real, F.imag)
ax3.scatter([0, 0], [-10, 10], s = 1)

fig.show()














