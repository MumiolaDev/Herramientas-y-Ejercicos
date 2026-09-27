"""La colisión MRT: (1) con todas las tasas iguales debe ser BGK exacto
—comparado contra la fórmula BGK escrita aparte, no contra el mismo
código—; (2) la viscosidad que produce debe ser ν = (τ − 1/2)/3, medida
con el decaimiento de una onda de corte, también a τ = 0.52 donde BGK ya
no es estable en las escenas bifásicas."""

from __future__ import annotations

import numpy as np
import pytest

from aguacero.lattice import EX, EY, W3, E, W, equilibrio, propagar
from aguacero.mundo import Mundo, Parametros


def _paso_bgk_explicito(m: Mundo) -> None:
    """Un paso BGK + Guo + Li escrito directamente sobre las poblaciones,
    sin momentos (dominio periódico sin paredes, isotérmico)."""
    tau = m.p.tau
    f = m.f
    rho = f.sum(axis=0)
    Fx, Fy, psi = m._fuerza_cohesion(rho)
    Fyt = Fy + rho * m.p.gravedad
    ux = ((f * EX).sum(axis=0) + 0.5 * Fx) / rho
    uy = ((f * EY).sum(axis=0) + 0.5 * Fyt) / rho
    eu = EX * ux + EY * uy
    guo = (1 - 0.5 / tau) * W3 * (3 * ((EX - ux) * Fx + (EY - uy) * Fyt) + 9 * eu * (EX * Fx + EY * Fyt))
    Q = m.p.sigma_li * (Fx**2 + Fy**2) / (np.maximum(psi**2, 1e-30) * tau)
    li = (3.0 * W * (3.0 * (E**2).sum(axis=1) - 2.0))[:, None, None] * Q
    m.f = propagar(f - (f - equilibrio(rho, ux, uy)) / tau + guo + li)


def test_mrt_con_tasas_iguales_es_bgk():
    params = dict(tau=0.8, gravedad=3e-5, isotermico=True)
    a = Mundo(50, 40, Parametros(colision="bgk", **params), marco=False, motor="numpy")
    b = Mundo(50, 40, Parametros(colision="bgk", **params), marco=False, motor="numpy")
    fila, col = np.mgrid[0:40, 0:50]
    for m in (a, b):
        m.agregar_liquido(np.hypot(col - 25, fila - 20) < 9)
    for _ in range(200):
        a.paso(1)
        _paso_bgk_explicito(b)
    assert np.abs(a.f - b.f).max() < 1e-13


@pytest.mark.parametrize("tau", [0.52, 0.8])
def test_viscosidad_por_decaimiento_de_onda_de_corte(tau):
    """u_x(y) = U sin(ky) en un líquido uniforme decae como exp(−νk²t).
    Con densidad uniforme la fuerza de cohesión es nula, así que esto mide
    sólo la parte viscosa de la colisión."""
    N = 64
    m = Mundo(4, N, Parametros(tau=tau, gravedad=0.0, isotermico=True), marco=False, motor="numpy")
    k = 2 * np.pi / N
    y = np.arange(N)[:, None] * np.ones((1, 4))
    rho = np.full((N, 4), m.rho_liquido)
    m.f = equilibrio(rho, 1e-3 * np.sin(k * y), np.zeros_like(rho))

    def amplitud():
        m.paso(1)
        return 2 * (m.ux[:, 0] * np.sin(k * y[:, 0])).mean()

    for _ in range(50):  # deja pasar el transitorio de la parte no-equilibrio inicial
        amplitud()
    a0, t0 = amplitud(), m.pasos
    for _ in range(1500):
        a1 = amplitud()
    nu_medida = -np.log(a1 / a0) / ((m.pasos - t0) * k**2)
    assert nu_medida == pytest.approx(m.p.viscosidad, rel=2e-3)
