"""Ejemplo 2D: billar de Sinai — un billar cuadrado con un disco central
removido. Es el ejemplo canónico de billar caóticamente disperso: la
trayectoria clásica que rebota contra el disco es sensible a condiciones
iniciales incluso sin ninguna otra irregularidad en la geometría (a
diferencia del billar rectangular vacío, que es integrable/separable).

Un paquete gaussiano lanzado desde una esquina choca con el disco, se
difracta a su alrededor, y tras rebotar unas pocas veces contra las
paredes desarrolla un patrón de "moteado" (speckle) irregular que llena
la cavidad de manera aproximadamente uniforme — no vuelve a parecerse a
una bolita rebotando limpiamente. Esa es la firma cuántica de la ergodicidad
clásica del billar de Sinai (conjetura de Berry: los autoestados de alta
energía de un sistema clásicamente caótico se comportan, localmente, como
una superposición de ondas planas con fases aleatorias).

Corre:
    python examples/billar_cuantico.py

Genera:
    salidas/billar_cuantico.gif
    salidas/billar_cuantico_ascii.txt
"""

from __future__ import annotations

import os
import time

from colapsoscopio import Grid2D, Simulation2D, SimulationConfig2D
from colapsoscopio.quantum.initial_conditions import GaussianPacket2D
from colapsoscopio.quantum.potentials import SinaiBilliard
from colapsoscopio.quantum.visualization import AsciiAnimator2D, MatplotlibAnimator2D


def main() -> None:
    # N+1=384=2^7*3: la transformada seno discreta (DST) usa internamente
    # una FFT de tamaño ~2(N+1), así que el tamaño de la malla importa para
    # el costo, no solo el número de puntos — un tamaño mal factorizado
    # cercano a este llegó a medir 5x más lento por punto (ver la
    # conversación que acompaña este commit).
    grid = Grid2D(x_min=-6.0, x_max=6.0, n_x=383, y_min=-6.0, y_max=6.0, n_y=383, boundary="dirichlet")
    potential = SinaiBilliard(v0_obstaculo=300.0, centro=(0.0, 0.0), radio=1.5)
    ic = GaussianPacket2D(x0=-4.0, y0=-4.0, sigma_x=0.5, sigma_y=0.5, kx0=6.0, ky0=5.0)

    config = SimulationConfig2D(
        grid=grid, potential=potential, initial_condition=ic, dt=1.5e-4, n_steps=18750, guardar_cada=125
    )

    print(f"malla: {grid.n_x}x{grid.n_y} = {grid.n_x * grid.n_y} puntos, dx={grid.dx:.5f}")
    t0 = time.time()
    traj = Simulation2D(config).run()
    print(f"tiempo de cómputo: {time.time() - t0:.1f}s")

    print(f"norma inicial={traj.norma[0]:.8f}  norma final={traj.norma[-1]:.8f}")
    print(f"<E> inicial={traj.energia[0]:.5f}  <E> final={traj.energia[-1]:.5f}")

    os.makedirs("salidas", exist_ok=True)

    animador = MatplotlibAnimator2D(traj, potencial_xy=potential(*grid.meshgrid()))
    animador.guardar("salidas/billar_cuantico.gif", fps=20)
    print("guardado: salidas/billar_cuantico.gif")

    ascii_animador = AsciiAnimator2D(traj)
    ascii_animador.guardar_texto("salidas/billar_cuantico_ascii.txt")
    print("guardado: salidas/billar_cuantico_ascii.txt")


if __name__ == "__main__":
    main()
