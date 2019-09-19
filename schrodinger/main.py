import numpy as np
import matplotlib.pyplot as plt
import math

import matplotlib.animation as animation
from matplotlib import style
style.use('ggplot')

e = math.e
PI = math.pi
hbar = 1.0
m = 1.9

def exp(x):
    return e**x

class Schrodinger:
    def __init__(self, x, phi, V):
        assert len(x) > 1
        assert len(x) == len(phi)
        assert len(x) == len(V)
        
        self.x = x.astype(complex)
        self.phi = phi.astype(complex)
        self.V = V.astype(complex)

        self.dx = x[1]-x[0]
        self.N = len(x)
        self.dk = 2 *  PI / (self.N * self.dx)
        self.k0 = -0.5 * self.N * self.dk

        self.k = self.k0 + self.dk * np.arange(self.N).astype(complex)

        self.dt = 0.05


    def fft(self, y):
        F = np.array(y).astype(complex)
        F *= self.dx / (2 * PI)**0.5
        phase = np.exp(-1j * self.x * self.k[0])
        F = F * phase
        F = np.fft.fft(F)
        F = F * np.exp(1j * self.x[0] * self.dk * np.array(range(self.N)))
        return F

    def ifft(self, F):
        y = np.array(F)
        y = y * np.exp(-1j * self.x[0] * self.dk * np.array(range(self.N)))
        y = np.fft.ifft(y)
        phase = np.exp(1j * self.x * self.k[0])
        y = y * phase
        y *= (2*PI)**0.5 / self.dx
        return y

    def evolve(self):
        self.phi = self.phi * np.exp(-1j * self.V * self.dt/2 * 1/hbar)

        p = self.fft(self.phi)
        p = p * np.exp(-1j * hbar * self.k * self.k * self.dt * 1/(2*m))
        self.phi = self.ifft(p)

        self.phi = self.phi * np.exp(-1j * self.V * self.dt/2 * 1/hbar)



##TESTING

dx = 0.01
x = np.arange(-100, 100, dx).astype(complex)

k0 = 10
phi = np.exp(-x*x)
phi = phi * np.exp(1j * x * k0)

V0 = 1e9
V = np.zeros(len(x))
for i in range(9000):
    V[-1-i] = V0
for i in range(8000):
    V[-1-i] = 0


S = Schrodinger(x, phi, V)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)


def animate(i):
    global S

    S.evolve()

    phi = S.phi
    p = S.fft(S.phi)

    
    ax1.clear()
    ax2.clear()
    ax1.scatter([0], [2], s = 0.1)
    ax2.scatter([0], [2], s = 0.1)

    ax1.plot(x.real, np.abs(phi))
    ax1.plot(x.real, 2*S.V.real/V0)
    ax2.plot(S.k.real, np.abs(p))

    ax2.set_xlabel('$k$')
    ax2.set_ylabel(r'$|\psi(k)|$')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel(r'$|\psi(x)|$')

    ax1.set_title("Position Space")
    ax2.set_title("Momentum Space")
    


ani = animation.FuncAnimation(fig, animate, interval=30)
fig.show()
