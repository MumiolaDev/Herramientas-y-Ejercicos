import numpy as np
import matplotlib.pyplot as plt


def f1(z,C1,C2, gama):
    tmp = 1j*C2 + 1j*gama*C1*np.abs(C1)**2
    return tmp

def f2(z,C1,C2, gama):
    tmp = 1j*C1 + 1j*C2*gama*np.abs(C2)**2
    return tmp

def rk4_step(z, C1, C2, h, gama):
    k1_C1 = h * f1(z, C1, C2, gama)
    k1_C2 = h * f2(z, C1, C2, gama)

    k2_C1 = h * f1(z + h/2, C1 + k1_C1/2, C2 + k1_C2/2, gama)
    k2_C2 = h * f2(z + h/2, C1 + k1_C1/2, C2 + k1_C2/2, gama)

    k3_C1 = h * f1(z + h/2, C1 + k2_C1/2, C2 + k2_C2/2, gama)
    k3_C2 = h * f2(z + h/2, C1 + k2_C1/2, C2 + k2_C2/2, gama)

    k4_C1 = h * f1(z + h, C1 + k3_C1, C2 + k3_C2, gama)
    k4_C2 = h * f2(z + h, C1 + k3_C1, C2 + k3_C2, gama)

    C1_sgte = C1 + (k1_C1 + 2*k2_C1 + 2*k3_C1 + k4_C1) / 6.
    C2_sgte = C2 + (k1_C2 + 2*k2_C2 + 2*k3_C2 + k4_C2) / 6.

    return C1_sgte, C2_sgte

## Condiciones iniciales
z0 = 0
zf = 50
h = 0.1

gammas = [0,2,4,6]
# En z=0
c1 = 1
c2 = 0

#### Integracion

paso = int((zf-z0)/h)

Z = np.arange(z0,zf+h,h)
N = len(Z)

fig, ax = plt.subplots(3,4)
n=0
for gamma in gammas:
    C1 = np.zeros(N, dtype=np.complex128)
    C2 = np.zeros(N, dtype=np.complex128)

    C1[0] = c1
    C2[0] = c2
    for i in range(1,N):
        C1[i], C2[i] = rk4_step(Z[i-1], C1[i-1], C2[i-1],h, gamma)

    ax[0,n].plot(Z, np.abs(C1)**2)
    ax[0,n].set_title("gamma ="+str(gamma))
    #ax[0,n].label_outer()

    ax[1,n].plot(Z, np.abs(C2)**2)
    #ax[1,n].set_title("gamma="+str(gamma))
    #ax[1,n].label_outer()

    ax[2,n].plot(Z, (np.abs(C1)**2+np.abs(C2)**2))
    #ax[2,n].set_title("gamma="+str(gamma))
    #ax[2,n].label_outer()

    n += 1


ax[0,0].set_ylabel("C1^2")
ax[1,0].set_ylabel("C2^2")
ax[2,0].set_ylabel("C1^2+C2^2")

plt.show()