import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

## EJEMPLO DE PLOTEAR LOS POLINOMIOS DE
## LEGENDRE. PUEDO USARLOS PA PLOTEAR MIS SOLUCIONES

N = 100
xvals = np.linspace(1,10,N)

x_1 = np.linspace(-1,1,N)

def Legendre(n,x):
    leg = sp.legendre(n)
    P_n = leg(np.cos(x))
    return P_n

def Bessel_J(m,x):
    bes = sp.jv(m,x)
    return bes
    
def Bessel_N(m,x):
    bes = sp.yn(m,x)
    return bes

def Bessel_I(m,x):
    bes = sp.iv(m,x)
    return bes

def Bessel_K(m,x):
    bes = sp.kv(m,x)
    return bes

def LeguendreAsociadas(l,m,x):
    leg = sp.lpmv(l,m,x)
    return leg

f1 = []
f2 = []
f3 = []
f4 = []
f5 = []
for i in range(3):
    f1 = Legendre(i, xvals)
    f2 = Bessel_J(i, xvals)
    f3 = Bessel_N(i, xvals)
    f4 = Bessel_I(i, xvals)
    f5 = Bessel_K(i, xvals)

    #plt.plot(xvals,f2, "+")
    #plt.plot(xvals,f3, "x")

l = 2
for m in [-2,-1,0,1,2]:
    f = LeguendreAsociadas(m,l, x_1)
    plt.plot(x_1,f)


plt.title("Polinomios")
plt.grid()
plt.legend()
plt.show()
