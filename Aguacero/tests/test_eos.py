"""La EOS por sí sola: punto crítico, identidad termodinámica de la
energía interna, y que ψ reproduce exactamente p_EOS."""

from __future__ import annotations

import numpy as np
import pytest

from aguacero.eos import G_PSEUDOPOTENCIAL, CarnahanStarling
from aguacero.lattice import CS2

eos = CarnahanStarling()


def test_punto_critico_es_inflexion_horizontal():
    rc, Tc, h = eos.rho_critica, eos.T_critica, 1e-4
    dp = (eos.presion(rc + h, Tc) - eos.presion(rc - h, Tc)) / (2 * h)
    d2p = (eos.presion(rc + h, Tc) - 2 * eos.presion(rc, Tc) + eos.presion(rc - h, Tc)) / h**2
    escala = eos.presion(rc, Tc) / rc
    # 0.3773 y 0.5218 son constantes redondeadas a 4 cifras: se exige eso
    assert abs(dp) / escala < 2e-3
    assert abs(d2p) * rc / escala < 2e-2


def test_energia_interna_cumple_la_identidad_termodinamica():
    """(∂u/∂v)_T = T(∂p/∂T)_v − p, con u = c_v T − aρ por unidad de masa y v = 1/ρ."""
    for rho, T in [(0.02, 0.07), (0.3, 0.066), (0.15, 0.1)]:
        h = 1e-6
        u = lambda r: eos.densidad_energia_interna(r, T) / r
        v = lambda r: 1.0 / r
        du_dv = (u(rho + h) - u(rho - h)) / (v(rho + h) - v(rho - h))
        derecha = T * eos.dp_dT(rho) - eos.presion(rho, T)
        assert du_dv == pytest.approx(derecha, rel=1e-6)


def test_pseudopotencial_reproduce_la_ecuacion_de_estado():
    rho = np.linspace(0.005, 0.45, 50)
    T = 0.7 * eos.T_critica
    psi = eos.psi(rho, T)
    p_red = CS2 * rho + G_PSEUDOPOTENCIAL * CS2 / 2 * psi**2
    assert p_red == pytest.approx(eos.presion(rho, T), rel=1e-12, abs=1e-15)


def test_maxwell_iguala_potencial_quimico():
    """La construcción de Maxwell (igualdad de áreas) debe dar igual p e
    igual energía libre de Gibbs por partícula en ambas fases."""
    T = 0.75 * eos.T_critica
    rg, rl = eos.coexistencia(T, epsilon=None)
    assert eos.presion(rg, T) == pytest.approx(eos.presion(rl, T), rel=1e-7)
    # μ = ∫ dp/ρ a lo largo de la isoterma (válido incluso a través del lazo)
    from scipy.integrate import quad

    dp_drho = lambda r: (eos.presion(r + 1e-7, T) - eos.presion(r - 1e-7, T)) / 2e-7
    delta_mu = quad(lambda r: dp_drho(r) / r, rg, rl, limit=400)[0]
    assert abs(delta_mu) < 1e-6
