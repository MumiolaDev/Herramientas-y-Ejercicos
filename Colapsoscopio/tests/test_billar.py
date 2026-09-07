"""EmptyBilliard es al billar rectangular vacío lo que InfiniteWell es al
pozo 1D: el único potencial 2D de este proyecto con autoestados analíticos
(producto de dos senos, uno por eje, porque el problema es separable), así
que la validación es la misma que en 1D — arrancar exactamente en un
autoestado y comprobar que |Psi|^2 queda estacionaria a precisión de
máquina, con la fase correcta.

SinaiBilliard (el mismo rectángulo con un disco central removido) no es
separable — es el ejemplo canónico de billar caóticamente disperso — así
que ahí no hay autoestado analítico contra el cual comparar; la validación
es la misma de siempre (conservación de norma/energía) más un chequeo
físico directo: la densidad dentro del disco debe quedar consistentemente
casi nula, porque V ahí es alto.
"""

from __future__ import annotations

import numpy as np
import pytest

from colapsoscopio import Grid2D, Simulation2D, SimulationConfig2D
from colapsoscopio.quantum.core.hamiltonian2d import Hamiltonian2D
from colapsoscopio.quantum.initial_conditions import GaussianPacket2D
from colapsoscopio.quantum.initial_conditions.eigenstate import Eigenstate
from colapsoscopio.quantum.potentials import EmptyBilliard, SinaiBilliard
from colapsoscopio.quantum.solvers.split_step_2d import SplitStepSolver2D


@pytest.mark.parametrize("n,m", [(1, 1), (2, 1), (1, 3), (2, 2)])
def test_autoestado_billar_vacio_es_estacionario(n, m):
    grid = Grid2D(x_min=0.0, x_max=10.0, n_x=64, y_min=0.0, y_max=8.0, n_y=64, boundary="dirichlet")
    potential = EmptyBilliard()
    wf, energia_analitica = potential.autoestado(n, m, grid)
    hamiltonian = Hamiltonian2D(grid=grid, potential=potential)
    solver = SplitStepSolver2D(hamiltonian)

    densidad_inicial = wf.densidad().copy()
    psi = wf
    for _ in range(300):
        psi = solver.paso(psi, 1e-3)

    assert psi.densidad() == pytest.approx(densidad_inicial, abs=5e-4)
    energia_numerica = hamiltonian.valor_esperado_energia(psi)
    assert energia_numerica == pytest.approx(energia_analitica, rel=1e-6)


def test_norma_y_energia_se_conservan_en_billar_sinai():
    grid = Grid2D(x_min=-6.0, x_max=6.0, n_x=128, y_min=-6.0, y_max=6.0, n_y=128, boundary="dirichlet")
    potential = SinaiBilliard(v0_obstaculo=200.0, centro=(0.0, 0.0), radio=1.5)
    ic = GaussianPacket2D(x0=-3.5, y0=2.5, sigma_x=0.6, sigma_y=0.6, kx0=5.0, ky0=-2.0)
    config = SimulationConfig2D(
        grid=grid, potential=potential, initial_condition=ic, dt=2e-4, n_steps=1500, guardar_cada=150
    )
    traj = Simulation2D(config).run()

    assert traj.norma == pytest.approx(1.0, abs=1e-8)
    drift_relativo = abs(traj.energia[-1] - traj.energia[0]) / abs(traj.energia[0])
    assert drift_relativo < 1e-3


def test_densidad_dentro_del_obstaculo_permanece_despreciable():
    grid = Grid2D(x_min=-6.0, x_max=6.0, n_x=128, y_min=-6.0, y_max=6.0, n_y=128, boundary="dirichlet")
    potential = SinaiBilliard(v0_obstaculo=200.0, centro=(0.0, 0.0), radio=1.5)
    ic = GaussianPacket2D(x0=-3.5, y0=2.5, sigma_x=0.6, sigma_y=0.6, kx0=5.0, ky0=-2.0)
    config = SimulationConfig2D(
        grid=grid, potential=potential, initial_condition=ic, dt=2e-4, n_steps=1500, guardar_cada=150
    )
    traj = Simulation2D(config).run()

    X, Y = grid.meshgrid()
    dentro_del_disco = X**2 + Y**2 <= potential.radio**2
    for i in range(traj.n_snapshots):
        p_en_disco = grid.integrate(traj.densidad(i) * dentro_del_disco)
        assert p_en_disco < 1e-3
