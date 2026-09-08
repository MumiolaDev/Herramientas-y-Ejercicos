"""Ejemplo: péndulo doble en el régimen *caótico* (energía alta, ángulos
iniciales grandes, cerca de poder invertirse). El retrato de fases deja de
ser una curva ordenada y pasa a llenar una región del espacio de fases de
forma irregular — sin cruzarse consigo mismo dos veces con la misma
tangente (eso violaría el determinismo), pero sin ningún patrón periódico
visible tampoco.

Corre:
    python examples/pendulo_doble_caotico.py

Genera:
    salidas/pendulo_doble_caotico.gif
"""

from __future__ import annotations

import os

import numpy as np

from caoscopio import DoublePendulum, Simulation, SimulationConfig
from caoscopio.visualization import PendulumAnimator


def main() -> None:
    sistema = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
    estado_inicial = np.array([2.4, 0.0, 2.4, 0.0])  # ~137°, energía alta

    config = SimulationConfig(
        sistema=sistema, estado_inicial=estado_inicial, dt=1e-3, n_steps=20000, guardar_cada=60
    )
    traj = Simulation(config).run()

    print(f"energía inicial={traj.energia[0]:.6f}  energía final={traj.energia[-1]:.6f}")
    drift = abs(traj.energia[-1] - traj.energia[0]) / abs(traj.energia[0])
    print(f"deriva relativa de energía: {drift:.2e}")

    os.makedirs("salidas", exist_ok=True)
    animador = PendulumAnimator(sistema, traj)
    animador.guardar("salidas/pendulo_doble_caotico.gif", fps=30)
    print("guardado: salidas/pendulo_doble_caotico.gif")


if __name__ == "__main__":
    main()
