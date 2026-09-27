"""Ley de Laplace en 2D: Δp = σ/R. Tres gotas de distinto radio, sin
gravedad, isotérmicas. Se exige una recta que pase por el origen — la
tensión superficial no es un parámetro del modelo, EMERGE de la fuerza
entre vecinos, así que esto prueba que la interfaz se comporta como una
interfaz de verdad."""

from __future__ import annotations

import numpy as np

from aguacero.mundo import Mundo, Parametros


def test_ley_de_laplace():
    puntos = []
    for R in (12, 16, 22):
        N = 90
        m = Mundo(N, N, Parametros(T_reducida=0.7, gravedad=0.0, isotermico=True), marco=False)
        fila, col = np.mgrid[0:N, 0:N]
        r = np.hypot(col - N / 2 + 0.5, fila - N / 2 + 0.5)
        m.agregar_liquido(r < R)
        m.paso(8000)
        p = m.presion()
        rho_in, rho_out = m.rho[r < R / 3].mean(), m.rho[r > R + 12].mean()
        R_ef = np.sqrt(((m.rho - rho_out) / (rho_in - rho_out)).sum() / np.pi)
        puntos.append((1 / R_ef, p[r < R / 3].mean() - p[r > R + 12].mean()))

    x, y = np.array(puntos).T
    (sigma, intercepto), *_ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)
    residuo = y - (sigma * x + intercepto)
    assert 5e-3 < sigma < 7.5e-3  # medido: 6.2e-3 a 0.7 T_c
    assert abs(intercepto) < 0.08 * y.max()
    assert np.abs(residuo).max() < 0.02 * y.max()
