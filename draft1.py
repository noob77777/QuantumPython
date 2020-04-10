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

def gce(t):
    z = 0+1j
    return e**(z*t)

x = np.arange(-100, 100, 0.1).astype(complex)
n = len(x)
k = np.array(x)
#for i in range(n):
#    k[i] = i*1.0/n*2*PI

def getabs(x):
    y = []
    for i in range(len(x)):
        y.append(abs(x[i]))
    y = np.array(y)
    return y

def Y(g, k):
    global x
    y = g(x)
    for i in range(len(y)):
        y[i] = y[i]*gce(k*x[i])
    return y

def f(x):
    y = np.array(x)
    for i in range(len(x)):
        y[i] = e**(-x[i]*x[i])
    return y

phi = Y(f, 0)
p = np.fft.fft(phi)

##t = 10000000

##for i in range(len(p)):
##    p[i] = p[i]*gce(-t*HH*k[i]*k[i])

##phi = np.fft.ifft(p)
##plt.scatter(x, getabs(phi), s = 1)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

t = 0
def animate(i):
    global t
    for i in range(len(p)):
        p[i] = p[i]*gce(-t*HH*k[i]*k[i])

    phi = np.fft.ifft(p)

    t += 10
    ##print(t)
    
    ax1.clear()
    plt.scatter([0], [2], s = 0.1)
    plt.plot(x, getabs(phi))


ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()



