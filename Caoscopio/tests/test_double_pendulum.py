"""Dos validaciones independientes para el péndulo doble (ver el docstring
de systems/double_pendulum.py para por qué hace falta más que la
conservación de energía):

1. El límite m2→0 debe reducirse exactamente a un péndulo simple de
   longitud l1 en θ1 — es una prueba de que las ecuaciones de movimiento
   están bien transcritas, no solo de que son autoconsistentes.
2. Conservación de energía (aproximada, ver core/integrator.py) tanto en
   el régimen regular (energía baja, ángulos pequeños) como en el caótico
   (energía alta) — el RK4 no tiene ninguna garantía estructural de
   conservarla, así que vale la pena chequear ambos regímenes por
   separado.
"""

from __future__ import annotations

import numpy as np
import pytest

from caoscopio import DoublePendulum, Simulation, SimulationConfig, SimplePendulum


def test_limite_masa_nula_reduce_a_pendulo_simple():
    l1, g = 1.3, 9.81
    doble = DoublePendulum(m1=1.0, m2=1e-8, l1=l1, l2=0.5, g=g)
    simple = SimplePendulum(l=l1, g=g)

    theta0, omega0 = 0.6, 0.0
    config_doble = SimulationConfig(
        sistema=doble, estado_inicial=np.array([theta0, omega0, theta0, omega0]),
        dt=1e-3, n_steps=5000, guardar_cada=50,
    )
    config_simple = SimulationConfig(
        sistema=simple, estado_inicial=np.array([theta0, omega0]), dt=1e-3, n_steps=5000, guardar_cada=50
    )
    traj_doble = Simulation(config_doble).run()
    traj_simple = Simulation(config_simple).run()

    theta1_doble = traj_doble.estados[:, 0]
    theta_simple = traj_simple.estados[:, 0]
    assert theta1_doble == pytest.approx(theta_simple, abs=1e-6)


def test_energia_se_conserva_regimen_regular():
    sistema = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
    config = SimulationConfig(
        sistema=sistema, estado_inicial=np.array([0.3, 0.0, 0.3, 0.0]),
        dt=1e-3, n_steps=20000, guardar_cada=100,
    )
    traj = Simulation(config).run()
    drift_relativo = abs(traj.energia[-1] - traj.energia[0]) / abs(traj.energia[0])
    assert drift_relativo < 1e-6


def test_energia_se_conserva_regimen_caotico():
    sistema = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
    config = SimulationConfig(
        sistema=sistema, estado_inicial=np.array([2.4, 0.0, 2.4, 0.0]),
        dt=1e-3, n_steps=20000, guardar_cada=100,
    )
    traj = Simulation(config).run()
    drift_relativo = abs(traj.energia[-1] - traj.energia[0]) / abs(traj.energia[0])
    assert drift_relativo < 1e-6
