"""Ejemplo: paquete de onda gaussiano rebotando dentro de un pozo de
potencial infinito ("partícula en una caja").

Corre:
    python examples/pozo_infinito.py

Genera:
    salidas/pozo_infinito.gif        (animación con matplotlib)
    salidas/pozo_infinito_ascii.txt  (misma trayectoria como "GIF" de texto)
"""

from __future__ import annotations

import os

from colapsoscopio import Grid1D, Simulation, SimulationConfig
from colapsoscopio.initial_conditions import GaussianPacket
from colapsoscopio.potentials import InfiniteWell
from colapsoscopio.visualization import AsciiAnimator, MatplotlibAnimator


def main() -> None:
    L = 20.0
    grid = Grid1D(x_min=0.0, x_max=L, n_points=512, boundary="dirichlet")
    potential = InfiniteWell()

    # paquete angosto cerca de la pared izquierda, con momento hacia +x:
    # debería viajar, rebotar contra la pared derecha, volver, etc.
    ic = GaussianPacket(x0=L * 0.3, sigma=L * 0.03, k0=6.0)

    config = SimulationConfig(
        grid=grid,
        potential=potential,
        initial_condition=ic,
        dt=2e-4,
        n_steps=15000,  # suficiente para ver al menos un rebote en la pared derecha
        guardar_cada=40,
    )
    traj = Simulation(config).run()

    print(f"norma inicial={traj.norma[0]:.8f}  norma final={traj.norma[-1]:.8f}")
    print(f"<E> inicial={traj.energia[0]:.5f}  <E> final={traj.energia[-1]:.5f}")

    os.makedirs("salidas", exist_ok=True)

    animador = MatplotlibAnimator(traj, potencial_x=potential(grid.x))
    animador.guardar("salidas/pozo_infinito.gif", fps=25)
    print("guardado: salidas/pozo_infinito.gif")

    ascii_animador = AsciiAnimator(traj)
    ascii_animador.guardar_texto("salidas/pozo_infinito_ascii.txt")
    print("guardado: salidas/pozo_infinito_ascii.txt")
    print("(para verlo animado en una terminal real:")
    print(" ascii_animador.reproducir(fps=15)  en vez de guardar_texto)")


if __name__ == "__main__":
    main()
