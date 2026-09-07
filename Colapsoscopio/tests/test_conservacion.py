"""El split-operator es unitario por construcción: la norma de Psi debe
conservarse a precisión de máquina (no solo "aproximadamente"), sea cual sea
la condición inicial o el potencial. Ese es el chequeo de sanidad más básico
y más importante del solver.
"""

from __future__ import annotations

import pytest

from colapsoscopio import Grid1D, Simulation, SimulationConfig
from colapsoscopio.quantum.initial_conditions import GaussianPacket
from colapsoscopio.quantum.potentials import HarmonicOscillator, InfiniteWell


def test_norma_se_conserva_en_pozo_infinito():
    grid = Grid1D(x_min=0.0, x_max=20.0, n_points=256, boundary="dirichlet")
    config = SimulationConfig(
        grid=grid,
        potential=InfiniteWell(),
        initial_condition=GaussianPacket(x0=6.0, sigma=0.6, k0=4.0),
        dt=1e-3,
        n_steps=300,
    )
    traj = Simulation(config).run()
    assert traj.norma == pytest.approx(1.0, abs=1e-9)


def test_norma_se_conserva_en_oscilador_armonico():
    grid = Grid1D(x_min=-15.0, x_max=15.0, n_points=512, boundary="periodic")
    config = SimulationConfig(
        grid=grid,
        potential=HarmonicOscillator(omega=1.0, mass=1.0),
        initial_condition=GaussianPacket(x0=2.0, sigma=1.0, k0=0.0),
        dt=1e-3,
        n_steps=300,
    )
    traj = Simulation(config).run()
    assert traj.norma == pytest.approx(1.0, abs=1e-9)


def test_energia_esperada_es_aproximadamente_constante():
    """<H> no se conserva exactamente con dt finito (el operador V/2-T-V/2
    no conmuta con H salvo en el límite dt->0), pero para dt lo bastante
    chico el drift debe ser pequeño frente a la propia escala de energía.
    """
    grid = Grid1D(x_min=-15.0, x_max=15.0, n_points=512, boundary="periodic")
    config = SimulationConfig(
        grid=grid,
        potential=HarmonicOscillator(omega=1.0, mass=1.0),
        initial_condition=GaussianPacket(x0=2.0, sigma=1.0, k0=0.0),
        dt=1e-3,
        n_steps=2000,
    )
    traj = Simulation(config).run()
    drift_relativo = abs(traj.energia[-1] - traj.energia[0]) / abs(traj.energia[0])
    assert drift_relativo < 1e-4
