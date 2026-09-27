"""Con densidad uniforme y u = 0 la ecuación de energía es la del calor.
Para el laplaciano de 5 puntos y un paso de Heun, el segundo momento de
un pulso crece EXACTAMENTE 2χ por paso y por eje (el término (χL)² de
Heun no aporta al segundo momento: Σx²L²T = Σ(Lx²)(LT) = 2ΣLT = 0).
Así que el test puede exigir precisión de máquina, no una tolerancia."""

from __future__ import annotations

import numpy as np
import pytest

from aguacero.mundo import Mundo, Parametros
from aguacero.termico import paso_temperatura


def test_varianza_crece_2_chi_por_paso():
    N = 80
    m = Mundo(N, N, Parametros(chi=0.1), marco=False)
    fila, col = np.mgrid[0:N, 0:N].astype(float)
    T = m.T0 + 1e-3 * np.exp(-((col - 40) ** 2 + (fila - 40) ** 2) / 8.0)
    rho = np.full((N, N), 0.3)
    cero = np.zeros_like(T)
    todo = np.ones((N, N), dtype=bool)
    nada = np.zeros((N, N), dtype=bool)

    def varianza_x(T):
        w = T - m.T0
        return (w * (col - 40) ** 2).sum() / w.sum()

    v0 = varianza_x(T)
    pasos = 60
    for _ in range(pasos):
        T = paso_temperatura(T, rho, cero, cero, todo, nada, m.T_fuente, m.p.eos, m.p.chi)
    assert varianza_x(T) - v0 == pytest.approx(2 * m.p.chi * pasos, rel=1e-9)
