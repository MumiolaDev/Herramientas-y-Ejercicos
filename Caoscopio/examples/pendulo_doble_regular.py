"""Ejemplo: péndulo doble en el régimen *regular* (energía baja, ángulos
iniciales pequeños). A diferencia del régimen caótico, el retrato de fases
acá es quasi-periódico: una curva que se enreda pero no llena el espacio
de fases de manera irregular — el contraste directo con
examples/pendulo_doble_caotico.py, mismo sistema, misma ley física, solo
distinta energía.

Corre:
    python examples/pendulo_doble_regular.py

Genera:
    salidas/pendulo_doble_regular.gif
"""

from __future__ import annotations

import os

import numpy as np

from caoscopio import DoublePendulum, Simulation, SimulationConfig
from caoscopio.visualization import PendulumAnimator


def main() -> None:
    sistema = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
    estado_inicial = np.array([0.3, 0.0, 0.3, 0.0])  # ~17°, energía baja

    config = SimulationConfig(
        sistema=sistema, estado_inicial=estado_inicial, dt=1e-3, n_steps=20000, guardar_cada=60
    )
    traj = Simulation(config).run()

    print(f"energía inicial={traj.energia[0]:.6f}  energía final={traj.energia[-1]:.6f}")
    drift = abs(traj.energia[-1] - traj.energia[0]) / abs(traj.energia[0])
    print(f"deriva relativa de energía: {drift:.2e}")

    os.makedirs("salidas", exist_ok=True)
    animador = PendulumAnimator(sistema, traj)
    animador.guardar("salidas/pendulo_doble_regular.gif", fps=30)
    print("guardado: salidas/pendulo_doble_regular.gif")


if __name__ == "__main__":
    main()
