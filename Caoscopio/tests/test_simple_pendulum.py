"""El péndulo simple es el sistema de control no caótico: 1 grado de
libertad, integrable, período conocido en el límite de pequeñas
oscilaciones. Eso da una validación independiente de la que ofrece la
conservación de energía (que solo prueba autoconsistencia, no que las
ecuaciones estén bien): comparar el período numérico contra la fórmula
T=2π√(l/g).
"""

from __future__ import annotations

import numpy as np
import pytest

from caoscopio import Simulation, SimulationConfig, SimplePendulum


def test_periodo_converge_a_pequenas_oscilaciones():
    l, g = 1.2, 9.81
    periodo_teorico = 2 * np.pi * np.sqrt(l / g)

    sistema = SimplePendulum(l=l, g=g)
    amplitud = 0.05  # rad, pequeña: régimen lineal
    config = SimulationConfig(
        sistema=sistema, estado_inicial=np.array([amplitud, 0.0]), dt=1e-4, n_steps=200000, guardar_cada=1
    )
    traj = Simulation(config).run()

    theta = traj.estados[:, 0]
    # cruces por cero descendentes de theta: cada dos cruces es un período completo
    cruces = np.where((theta[:-1] > 0) & (theta[1:] <= 0))[0]
    assert len(cruces) >= 3
    periodos = np.diff(traj.tiempos[cruces])
    periodo_numerico = periodos.mean()

    assert periodo_numerico == pytest.approx(periodo_teorico, rel=1e-3)


def test_energia_se_conserva():
    sistema = SimplePendulum(l=1.0, g=9.81)
    config = SimulationConfig(
        sistema=sistema, estado_inicial=np.array([2.5, 0.0]), dt=1e-3, n_steps=20000, guardar_cada=100
    )
    traj = Simulation(config).run()
    drift_relativo = abs(traj.energia[-1] - traj.energia[0]) / abs(traj.energia[0])
    assert drift_relativo < 1e-6
