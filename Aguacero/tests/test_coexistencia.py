"""La validación central del modelo de fluido: interfaces PLANAS en
equilibrio (sin curvatura, sin gravedad, isotérmicas) comparadas con la
condición teórica de equilibrio mecánico del pseudopotencial,

    ∫_{ρg}^{ρl} (p0 − p_EOS) ψ'/ψ^{1+ε} dρ = 0.

- σ = 0 (Guo puro): la teoría dice ε = 0 SIN parámetros libres.
- σ > 0 (corrección de Li): ε = κσ con κ medido una vez; el test exige
  que el MISMO κ funcione a dos temperaturas y prediga ambas densidades.
"""

from __future__ import annotations

import numpy as np
import pytest

from aguacero.mundo import KAPPA_LI, Mundo, Parametros


def _losa(T_reducida: float, sigma_li: float, pasos: int = 20000):
    m = Mundo(128, 4, Parametros(T_reducida=T_reducida, gravedad=0.0, sigma_li=sigma_li, isotermico=True), marco=False)
    _, col = np.mgrid[0:4, 0:128]
    m.agregar_liquido((col >= 32) & (col < 96))
    m.paso(pasos)
    return m, m.rho[0, 0], m.rho[0, 64]


def test_guo_puro_obedece_epsilon_cero_sin_parametros_libres():
    m, rg, rl = _losa(0.8, sigma_li=0.0)
    teo_g, teo_l = m.p.eos.coexistencia(m.T0, epsilon=0.0)
    assert rg == pytest.approx(teo_g, rel=1e-3)
    assert rl == pytest.approx(teo_l, rel=1e-4)


@pytest.mark.parametrize("T_reducida", [0.8, 0.9])
def test_correccion_de_li_desplaza_epsilon_linealmente(T_reducida):
    sigma = 0.2
    m, rg, rl = _losa(T_reducida, sigma_li=sigma)
    teo_g, teo_l = m.p.eos.coexistencia(m.T0, epsilon=KAPPA_LI * sigma)
    assert rg == pytest.approx(teo_g, rel=1e-2)
    assert rl == pytest.approx(teo_l, rel=1e-3)


def test_parametros_por_defecto_quedan_cerca_de_maxwell():
    m, rg, rl = _losa(0.7, sigma_li=Parametros().sigma_li)
    mx_g, mx_l = m.p.eos.coexistencia(m.T0, epsilon=None)
    assert rg == pytest.approx(mx_g, rel=0.02)
    assert rl == pytest.approx(mx_l, rel=2e-3)
