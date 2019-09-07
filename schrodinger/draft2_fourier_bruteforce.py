import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from matplotlib import style
style.use('ggplot')

PI = math.pi
e = math.e
h = 6.626e-34
m = 9.1e-31
HH = h/(4*PI*m)
START = -10.0
END = 10.0
STEP = 0.1

def ei(t):
    z = 0+1j
    return e**(z*t)

def f(x):
    return e**(-x*x)

def Y(x, k = 0):
    return f(x)*ei(k*x)


dp1 = {}
def p(k):
    if k in dp1:
        return dp1[k]
    
    cst = 1.0/(2*PI)**0.5
    y = 0
    x = np.arange(START, END, STEP)
    for xx in x:
        y += Y(xx)*ei(-k*xx)*STEP
    y *= cst
    dp1[k] = y
    return y

def phi(x, t):
    cst = 1.0/(2*PI)**0.5
    y = 0
    k = np.arange(START, END, STEP)
    for kk in k:
        y += p(kk)*ei(-kk*kk*HH*t)*ei(kk*x)*STEP
    y *= cst
    return y

T = 0
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
def animate(i):
    global T
    x = np.arange(START, END, STEP)
    y = np.arange(START, END, STEP).astype(complex)
    for i in range(len(x)):
        y[i] = phi(x[i], T)

    for i in range(len(x)):
        y[i] = abs(y[i])

    T += 1000
    ax1.clear()
    ax1.scatter([0], [10], s = 0.01)
    ax1.scatter(x, y, s = 1)
    print('fps')


ani = animation.FuncAnimation(fig, animate, interval = 1)
fig.show()
    
