import numpy as np
import matplotlib.pyplot as plt
import math

import matplotlib.animation as animation
from matplotlib import style
style.use('ggplot')

PI = math.pi
e = math.e
h = 6.626e-34
m = 9.1e-31
HH = h/(4*PI*m)

def ei(t):
    z = 0+1j
    return e**(z*t)

START = -100
END = 100
STEP = 0.1

x = np.arange(START, END, STEP).astype(complex)
n = len(x)

def f(x):
    y = np.array(x)
    for i in range(n):
##        if abs(x[i]) < 1 :
##            y[i] = 0.5
##        else:
##            y[i] = 0

        y[i] = e**(-(x[i]+50)*(x[i]+50))*ei(10*x[i]) + e**(-(x[i]-50)*(x[i]-50))*ei(-10*x[i])
        
    return y


def Y(g, k):
    y = g(x)
    for i in range(len(y)):
        y[i] = y[i]*ei(k*x[i])
    return y

phi = Y(f, 5)

def fft(f):
    y = np.array(f)
    y *= STEP/(2*PI)**0.5
    phase = np.exp(-1j*x*START/2*PI)
    y = y*phase
    F = np.fft.fft(y)
    F = F*np.exp(1j*START*STEP*np.array(range(n)))
    return F

def ifft(f):
    y = np.array(f)
    y = y*np.exp(-1j*START*STEP*np.array(range(n)))
    y = np.fft.ifft(y)
    phase = np.exp(1j*x*START/2*PI)
    y = y*phase
    y *= (2*PI)**0.5/STEP
    return y
    

p = fft(phi)
##plt.plot(x.real, np.abs(p))
##plt.show()
##plt.plot(x.real, np.abs(phi))
##plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(2,1,1, title = "Position Space")
ax2 = fig.add_subplot(2,1,2, title = "Momentum Space")



t = 0
def animate(i):
    global t
    for i in range(len(p)):
        p[i] = p[i]*ei(-t*HH*x[i]*x[i])

    phi = ifft(p)

    t += 1
    
    ax1.clear()
    ax2.clear()
    ax1.scatter([0], [2], s = 0.1)
    ax2.scatter([0], [2], s = 0.1)

    ax1.plot(x.real, np.abs(phi))
    ax1.plot(x.real, phi.real)
    ax2.plot(x.real, np.abs(p))
    ax2.plot(x.real, p.real)

    ax2.set_xlabel('$k$')
    ax2.set_ylabel(r'$|\psi(k)|$')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel(r'$|\psi(x)|$')

    ax1.set_title("Position Space")
    ax2.set_title("Momentum Space")


ani = animation.FuncAnimation(fig, animate, interval=30)
fig.show()






