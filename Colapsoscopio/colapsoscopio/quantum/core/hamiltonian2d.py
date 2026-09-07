"""Hamiltoniano H = T + V para una partícula en 2D. Ver hamiltonian.py (1D)
para la idea general — acá T = hbar^2(kx^2+ky^2)/2m se diagonaliza con FFT2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colapsoscopio.quantum.core.grid2d import Grid2D
from colapsoscopio.quantum.potentials.base2d import Potential2D


@dataclass
class Hamiltonian2D:
    grid: Grid2D
    potential: Potential2D
    hbar: float = 1.0
    mass: float = 1.0

    def energia_cinetica_k(self) -> np.ndarray:
        KX, KY = self.grid.meshgrid_k()
        return (self.hbar**2) * (KX**2 + KY**2) / (2 * self.mass)

    def potencial_xy(self) -> np.ndarray:
        X, Y = self.grid.meshgrid()
        return self.potential(X, Y)

    def valor_esperado_energia(self, psi) -> float:
        from colapsoscopio.quantum.core.state2d import WaveFunction2D

        assert isinstance(psi, WaveFunction2D)
        psi_k = self.grid.transformar_ida(psi.psi)
        densidad_k = np.abs(psi_k) ** 2
        densidad_k = densidad_k / np.sum(densidad_k)
        energia_cinetica = float(np.sum(self.energia_cinetica_k() * densidad_k))
        energia_potencial = psi.valor_esperado(self.potencial_xy())
        return energia_cinetica + energia_potencial
