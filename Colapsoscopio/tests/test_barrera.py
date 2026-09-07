"""La barrera no tiene autoestados analíticos (ver potentials/barrier.py), así
que no hay una fase exacta contra la cual comparar como en
test_autoestados.py. La validación disponible es doble: la conservación de
norma/energía (idéntica a cualquier otro potencial, porque no depende de que
el potencial sea soluble) y un chequeo *cualitativo* pero físicamente
central del efecto túnel: con energía media menor que la barrera, tiene que
aparecer probabilidad no despreciable del otro lado, y la mayoría tiene que
seguir siendo reflejada (si no, el régimen elegido no sería "túnel").
"""

from __future__ import annotations

import pytest

from colapsoscopio import Grid1D, Simulation, SimulationConfig
from colapsoscopio.quantum.initial_conditions import GaussianPacket
from colapsoscopio.quantum.potentials import PotentialBarrier


def _simular_barrera(n_steps=6000, guardar_cada=100):
    hbar = mass = 1.0
    barrera = PotentialBarrier(v0=1.0, centro=0.0, ancho=1.5)
    grid = Grid1D(x_min=-25.0, x_max=25.0, n_points=1024, boundary="periodic")
    ic = GaussianPacket(x0=-8.0, sigma=2.0, k0=1.0)
    config = SimulationConfig(
        grid=grid, potential=barrera, initial_condition=ic, dt=5e-4, n_steps=n_steps, guardar_cada=guardar_cada
    )
    return Simulation(config).run(), barrera, grid


def test_norma_y_energia_se_conservan_con_barrera():
    traj, _, _ = _simular_barrera()
    assert traj.norma == pytest.approx(1.0, abs=1e-9)
    drift_relativo = abs(traj.energia[-1] - traj.energia[0]) / abs(traj.energia[0])
    assert drift_relativo < 1e-4


def test_hay_tunelamiento_con_energia_menor_que_la_barrera():
    """Energía media del paquete (k0=1, sigma=2 -> <E>=0.531) por debajo de
    V0=1.0: clásicamente el paquete debería reflejarse por completo. Si el
    solver es correcto, una fracción apreciable debe aparecer del otro lado
    de la barrera de todas formas.
    """
    traj, barrera, grid = _simular_barrera(n_steps=36000, guardar_cada=1800)
    x = grid.x
    densidad_final = traj.densidad(-1)
    lado_derecho = x > (barrera.centro + barrera.ancho / 2)
    lado_izquierdo = x < (barrera.centro - barrera.ancho / 2)
    p_transmitida = grid.integrate(densidad_final * lado_derecho)
    p_reflejada = grid.integrate(densidad_final * lado_izquierdo)

    assert 0.05 < p_transmitida < 0.5, "se espera una fracción transmitida apreciable pero minoritaria"
    assert p_reflejada > p_transmitida, "con <E> < V0 la mayoría de la probabilidad debe reflejarse"
    assert p_transmitida + p_reflejada == pytest.approx(1.0, abs=0.02)
