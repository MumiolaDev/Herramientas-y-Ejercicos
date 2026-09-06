"""Hamiltoniano H = T + V para una partícula en 1D.

T = p^2 / (2m) se diagonaliza en el espacio de momentos (autovalores hbar^2 k^2
/ 2m); V(x) se diagonaliza en el espacio de posiciones. Esta separación es
exactamente la que explota el propagador split-step de Fourier: no
ensamblamos una matriz H completa en ningún momento (para un átomo de
hidrógeno en 3D eso sería intratable en memoria), solo evaluamos T y V donde
cada uno es diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colapsoscopio.core.grid import Grid1D
from colapsoscopio.potentials.base import Potential


@dataclass
class Hamiltonian1D:
    grid: Grid1D
    potential: Potential
    hbar: float = 1.0
    mass: float = 1.0

    def energia_cinetica_k(self) -> np.ndarray:
        """T(k) = hbar^2 k^2 / (2m), evaluado en la malla de momentos."""
        return (self.hbar**2) * (self.grid.k**2) / (2 * self.mass)

    def potencial_x(self) -> np.ndarray:
        """V(x), evaluado en la malla de posiciones."""
        return self.potential(self.grid.x)

    def valor_esperado_energia(self, psi) -> float:
        """<H> = <T> + <V>, calculado por separado en cada representación."""
        from colapsoscopio.core.state import WaveFunction

        assert isinstance(psi, WaveFunction)
        psi_k = self.grid.transformar_ida(psi.psi)
        densidad_k = np.abs(psi_k) ** 2
        densidad_k = densidad_k / np.sum(densidad_k)
        energia_cinetica = float(np.sum(self.energia_cinetica_k() * densidad_k))
        energia_potencial = psi.valor_esperado(self.potencial_x())
        return energia_cinetica + energia_potencial
