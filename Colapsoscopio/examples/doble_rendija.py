"""Ejemplo 2D: el experimento de la doble rendija con un paquete de onda.
Un paquete gaussiano incide sobre una pantalla opaca con dos aberturas; del
otro lado emerge el patrón de interferencia característico —franjas de
máximos y mínimos, no dos manchas separadas como predeciría la intuición
clásica de "partícula que pasa por una rendija u otra"—.

Corre:
    python examples/doble_rendija.py

Genera:
    salidas/doble_rendija.gif
    salidas/doble_rendija_ascii.txt
"""

from __future__ import annotations

import os

from colapsoscopio import Grid2D, Simulation2D, SimulationConfig2D
from colapsoscopio.quantum.initial_conditions import GaussianPacket2D
from colapsoscopio.quantum.potentials import DoubleSlit
from colapsoscopio.quantum.visualization import AsciiAnimator2D, MatplotlibAnimator2D


def main() -> None:
    grid = Grid2D(x_min=-20.0, x_max=20.0, n_x=256, y_min=-15.0, y_max=15.0, n_y=256)
    potential = DoubleSlit(v0=40.0, x_pantalla=0.0, grosor=0.5, separacion=3.0, ancho_rendija=0.8)

    # sigma_y grande (frente ancho en y): así el paquete ilumina ambas
    # rendijas de forma parecida a una onda plana, en vez de pasar
    # esencialmente por una sola.
    ic = GaussianPacket2D(x0=-8.0, y0=0.0, sigma_x=1.5, sigma_y=3.0, kx0=3.0, ky0=0.0)

    config = SimulationConfig2D(
        grid=grid, potential=potential, initial_condition=ic, dt=2e-3, n_steps=3000, guardar_cada=30
    )
    traj = Simulation2D(config).run()

    print(f"norma inicial={traj.norma[0]:.8f}  norma final={traj.norma[-1]:.8f}")
    print(f"<E> inicial={traj.energia[0]:.5f}  <E> final={traj.energia[-1]:.5f}")

    os.makedirs("salidas", exist_ok=True)

    animador = MatplotlibAnimator2D(traj, potencial_xy=potential(*grid.meshgrid()))
    animador.guardar("salidas/doble_rendija.gif", fps=20)
    print("guardado: salidas/doble_rendija.gif")

    ascii_animador = AsciiAnimator2D(traj)
    ascii_animador.guardar_texto("salidas/doble_rendija_ascii.txt")
    print("guardado: salidas/doble_rendija_ascii.txt")


if __name__ == "__main__":
    main()
