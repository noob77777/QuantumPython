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
HH = h/(4*m*PI)
H = h/(2*PI)
inf = 1e6

def ei(t):
    z = 0+1j
    return e**(z*t)

START = -10
END = 10
STEP = 0.1

x = np.arange(START, END, STEP).astype(complex)
n = len(x)

def f(x):
    y = np.array(x)
    for i in range(n):
        y[i] = math.sin(x[i])+math.sin(5*x[i])

##        y[i] = e**(-(x[i]+50)*(x[i]+50))*ei(10*x[i]) + e**(-(x[i]-50)*(x[i]-50))*ei(-10*x[i])
        
    return y


def Y(g, k):
    y = g(x)
    for i in range(len(y)):
        y[i] = y[i]*ei(k*x[i])
    return y

def V(x):
    y = np.array(x)
    for i in range(len(x)):
        y[i] = 0
##    for i in range(len(x)//2+100, len(x)):
##        y[i] = inf
##        y[i] = 0

        
##    for i in range(900):
##        y[i] = inf
##    for i in range(1100,1110):
##        y[i] = inf
    return y

def fft(f):
    y = np.array(f)
    y *= STEP/(2*PI)**0.5
    phase = np.exp(-1j*x*START*PI/2)
    y = y*phase
    F = np.fft.fft(y)
    F = F*np.exp(1j*START*STEP*np.array(range(n)))
    return F

def ifft(f):
    y = np.array(f)
    y = y*np.exp(-1j*START*STEP*np.array(range(n)))
    y = np.fft.ifft(y)
    phase = np.exp(1j*x*START*PI/2)
    y = y*phase
    y *= (2*PI)**0.5/STEP
    return y
    
phi = Y(f, 0)
p = fft(phi)

v = V(x)

def step_phi(dt):
    global phi
    H = 1
    phi = phi*np.exp(-1j*v*dt/H)

def step_p(dt):
    global p
    ##HH = 1e-4
    p = p*np.exp(-1j*HH*dt*x*x)





fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
#ax3 = fig.add_subplot(3, 1, 3)

#ax3.plot(x.real, v.real)

t = 25
def animate(i):
    global phi, p

    step_phi(t/2)

    p = fft(phi)
    step_p(t)
##    for i in range(len(p)):
##        p[i] = p[i]*ei(-t*HH*x[i]*x[i])

    phi = ifft(p)
    step_phi(t/2)
    
    ax1.clear()
    ax2.clear()
    ax1.scatter([0], [2], s = 0.1)
    ax2.scatter([0], [2], s = 0.1)

    ax1.plot(x.real, np.abs(phi))
    ax1.plot(x.real, 2*v.real/inf)
    #ax1.plot(x.real, phi.real)
    ax2.plot(x.real, np.abs(p))
    #ax2.plot(x.real, p.real)

    ax2.set_xlabel('$k$')
    ax2.set_ylabel(r'$|\psi(k)|$')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel(r'$|\psi(x)|$')

    ax1.set_title("Position Space")
    ax2.set_title("Momentum Space")
    ##print("f")
    


ani = animation.FuncAnimation(fig, animate, interval=1)
fig.show()
