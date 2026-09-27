from __future__ import annotations

import numpy as np
import pytest

from aguacero.mundo import Mundo, Parametros
from aguacero.termico import paso_temperatura


def _escena_completa(motor):
    m = Mundo(70, 50, motor=motor)
    fila, col = np.mgrid[0:50, 0:70]
    m.agregar_liquido(fila > 38)
    m.agregar_liquido(np.hypot(col - 35, fila - 18) < 7)
    m.agregar_pared((fila >= 28) & (fila < 30) & (col > 5) & (col < 25))
    m.agregar_fuente((fila == 48) & (col > 40) & (col < 60), 1.05)
    return m


@pytest.mark.parametrize("motor", ["numpy", "numba"])
def test_masa_se_conserva_a_precision_de_maquina(motor):
    """Colisión (conserva ρ localmente), propagación (una permutación) y
    rebote (otra permutación): la masa total sólo puede cambiar por
    redondeo. Con paredes, gravedad y calefacción activas."""
    if motor == "numba":
        pytest.importorskip("numba")
    m = _escena_completa(motor)
    M0 = m.masa_total()
    m.paso(1500)
    assert np.isfinite(m.f).all()
    assert abs(m.masa_total() - M0) / M0 < 1e-12


def test_conduccion_conserva_la_energia_termica_exactamente():
    """Sólo conducción (u = 0) sobre un perfil de densidad bifásico
    abrupto y con paredes adiabáticas: Σ ρ c_v T debe conservarse al
    redondeo (flujos antisimétricos por cara) y el esquema no debe
    explotar pese al contraste 40:1 de densidad (media armónica)."""
    m = Mundo(60, 40, Parametros(T_reducida=0.7))
    fila, col = np.mgrid[0:40, 0:60]
    rho = np.where(col < 30, m.rho_liquido, m.rho_vapor) * np.ones((40, 60))
    T = m.T0 * (1 + 0.3 * np.exp(-((col - 20) ** 2 + (fila - 20) ** 2) / 30.0))
    cero = np.zeros_like(T)
    sin_fuente = np.zeros_like(m.solido)
    fluido = m.fluido
    E0 = (rho * T)[fluido].sum()
    for _ in range(3000):
        T = paso_temperatura(T, rho, cero, cero, fluido, sin_fuente, m.T_fuente, m.p.eos, m.p.chi)
    assert np.isfinite(T).all()
    assert (rho * T)[fluido].sum() == pytest.approx(E0, rel=1e-12)
    # y relaja hacia el equilibrio: sin extremos nuevos (principio del máximo)
    assert T[fluido].max() <= m.T0 * 1.3 + 1e-12
    assert T[fluido].min() >= m.T0 - 1e-12
