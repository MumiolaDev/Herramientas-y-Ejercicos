import numpy as np
import matplotlib.pyplot as plt

## DEFINIENDO LA INTEGRACION POR METODO RUNGE KUTTA DE ORDEN 4
## INCLUYENDO GAMMA PARAMETRO DE CONTROL DE LA FUNCION f
def Rk4_integracion(f, z0, C0, zf, dx, gamma):

    num_pasos = int( (zf-z0)/dx )

    valores_z = np.linspace(z0,zf,num_pasos +1, dtype=np.complex128)
    valores_C = np.zeros( ( num_pasos+1, len(C0) ), dtype=np.complex128 )

    valores_C[0] = C0

    for i in range(num_pasos):
        z_ = valores_z[i]
        C_ = valores_C[i]

        k1 = f(z_,C_,gamma)
        k2 = f( z_+dx*0.5 , C_+k1*0.5*dx,gamma)
        k3 = f( z_+dx*0.5 , C_+k2*0.5*dx,gamma)
        k4 = f( z_+dx     , C_+k3*dx,    gamma)

        C_sgte = C_ + ( k1 + 2*k2 + 2*k3 + k4 )*dx/6.

        valores_C[i+1] = C_sgte
        
    return valores_z, valores_C

## FUNCION f( C(z) ) = dC/dz
## QUE DEFINE LA ECUACION DEIFERENCIAL
## GAMMA ES UN PARAMETRO
def f(z, C, gamma):
    N = len(C)
    valores_f = np.zeros(N, dtype=np.complex128)

    for n in range(1, N-1):
        valores_f[n] = 1j *(C[n+1]+C[n-1])*V  + 1j*gamma*C[n]*np.abs(C[n])**2
    
    return valores_f

## DEFINIENDO LAS CONDICIONES INICIALES
N = 101
nc = 50
V = 1
k = 0.9
A = 1.9
z0 = 0
zf = 45
dx = .01
C0 = np.zeros(N, dtype=np.complex128)
for n in range(nc-1, nc+2):
    C0[n] = A* np.exp( -1j*k* (n-nc))/(np.cosh( (A/np.sqrt(2.))*(n-nc) ))

## GRAFICAREMOS LA EVOLUCION DE C CON RESPECTO A z
## PARA 6 GAMMAS DISTINTOS.

fig, axs = plt.subplots(2,3)
gammas = [0, 0.1, 0.5, 0.7, 1, 1.5]
n=0
for i in range(2):
    for j in range(3):

        Z1, C1 = Rk4_integracion(f, z0, C0, zf, dx, gammas[n])
        C1_cuad =abs(C1)
        N1_mesh, Z1_mesh = np.meshgrid(range(N), Z1)
        N1_mesh =abs(N1_mesh)
        Z1_mesh = abs(Z1_mesh)

        axs[i,j].pcolormesh(N1_mesh, Z1_mesh, C1_cuad, cmap = "gray")
        axs[i,j].set_title("gamma="+str(gammas[n]))

        n+=1

plt.show()