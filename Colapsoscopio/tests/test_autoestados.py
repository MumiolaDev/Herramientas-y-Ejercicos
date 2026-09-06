"""Un autoestado del Hamiltoniano es estacionario: |Psi(x,t)|^2 debe quedar
exactamente igual a |Psi(x,0)|^2 en todo instante, y Psi solo debe acumular
la fase global e^{-i E_n t/hbar}. Comparar contra estos dos hechos analíticos
es la validación más fuerte disponible para el solver, porque no depende de
ningún otro cálculo numérico externo (los autoestados vienen en forma
cerrada).
"""

from __future__ import annotations

import numpy as np
import pytest

from colapsoscopio import Grid1D, Simulation, SimulationConfig
from colapsoscopio.initial_conditions import Eigenstate
from colapsoscopio.potentials import HarmonicOscillator, InfiniteWell


@pytest.mark.parametrize("n", [1, 2, 3])
def test_autoestado_pozo_infinito_es_estacionario(n):
    grid = Grid1D(x_min=0.0, x_max=10.0, n_points=400, boundary="dirichlet")
    ic = Eigenstate(n=n)
    config = SimulationConfig(
        grid=grid, potential=InfiniteWell(), initial_condition=ic, dt=1e-3, n_steps=400
    )
    traj = Simulation(config).run()

    densidad_inicial = traj.densidad(0)
    for i in range(traj.n_snapshots):
        assert traj.densidad(i) == pytest.approx(densidad_inicial, abs=5e-4)

    _assert_fase_global_correcta(traj, ic.energia)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_autoestado_oscilador_armonico_es_estacionario(n):
    omega, mass, hbar = 1.0, 1.0, 1.0
    l0 = (hbar / (mass * omega)) ** 0.5
    grid = Grid1D(x_min=-10 * l0, x_max=10 * l0, n_points=800, boundary="periodic")
    ic = Eigenstate(n=n)
    config = SimulationConfig(
        grid=grid,
        potential=HarmonicOscillator(omega=omega, mass=mass),
        initial_condition=ic,
        dt=1e-3,
        n_steps=400,
        hbar=hbar,
        mass=mass,
    )
    traj = Simulation(config).run()

    densidad_inicial = traj.densidad(0)
    for i in range(traj.n_snapshots):
        assert traj.densidad(i) == pytest.approx(densidad_inicial, abs=5e-4)

    _assert_fase_global_correcta(traj, ic.energia, hbar=hbar)


def _assert_fase_global_correcta(traj, energia_analitica: float, hbar: float = 1.0) -> None:
    """Psi(x,t)/Psi(x,0) debe ser (aprox.) el mismo número complejo
    e^{-i E t/hbar} para todo x donde Psi(x,0) no sea ~0."""
    psi0 = traj.estados[0]
    mascara = np.abs(psi0) > 0.1 * np.abs(psi0).max()

    for i in [traj.n_snapshots // 2, traj.n_snapshots - 1]:
        t = traj.tiempos[i]
        fase_esperada = np.exp(-1j * energia_analitica * t / hbar)
        razon = traj.estados[i][mascara] / psi0[mascara]
        assert razon == pytest.approx(fase_esperada, abs=5e-3)
