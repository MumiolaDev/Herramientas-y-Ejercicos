"""Ejemplo: péndulo simple — el caso de control *no caótico*. Un solo
grado de libertad, integrable: el retrato de fases es siempre una curva
cerrada que nunca se cruza consigo misma (para energía por debajo de la
separatriz que lleva a la vuelta completa).

Corre:
    python examples/pendulo_simple.py

Genera:
    salidas/pendulo_simple.gif
"""

from __future__ import annotations

import os

import numpy as np

from caoscopio import Simulation, SimulationConfig, SimplePendulum
from caoscopio.visualization import PendulumAnimator


def main() -> None:
    sistema = SimplePendulum(l=1.0, g=9.81)
    estado_inicial = np.array([2.2, 0.0])  # ~126°, oscilación grande pero no llega a invertirse

    config = SimulationConfig(
        sistema=sistema, estado_inicial=estado_inicial, dt=1e-3, n_steps=12000, guardar_cada=40
    )
    traj = Simulation(config).run()

    print(f"energía inicial={traj.energia[0]:.6f}  energía final={traj.energia[-1]:.6f}")
    drift = abs(traj.energia[-1] - traj.energia[0]) / abs(traj.energia[0])
    print(f"deriva relativa de energía: {drift:.2e}")

    os.makedirs("salidas", exist_ok=True)
    animador = PendulumAnimator(sistema, traj)
    animador.guardar("salidas/pendulo_simple.gif", fps=30)
    print("guardado: salidas/pendulo_simple.gif")


if __name__ == "__main__":
    main()
