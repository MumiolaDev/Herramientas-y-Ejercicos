"""Pozo de potencial infinito ("partícula en una caja").

Convención importante: V(x) = 0 en todo el interior. El confinamiento no se
modela con una función que "se dispara a infinito" en los bordes (eso no es
representable en punto flotante ni compatible con un método espectral) sino
con la condición de frontera del Grid1D: úsese siempre junto a
Grid1D(..., boundary="dirichlet"), que ya impone Psi(x_min) = Psi(x_max) = 0
exactamente. El propio solver, al trabajar en la base seno (autofunciones de
-d^2/dx^2 con Dirichlet), hereda ese confinamiento sin necesidad de un
potencial de pared.
"""

from __future__ import annotations

import numpy as np

from colapsoscopio.potentials.base import Potential


class InfiniteWell(Potential):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def autoestado(self, n: int, grid, hbar: float = 1.0, mass: float = 1.0):
        """Autoestado n-ésimo exacto (n=1,2,3,... — n=1 es el estado
        fundamental), válido para grid.boundary == "dirichlet":

            Psi_n(x) = sqrt(2/L) sin(n*pi*(x - x_min)/L)
            E_n = n^2 pi^2 hbar^2 / (2 m L^2)
        """
        from colapsoscopio.core.state import WaveFunction

        if grid.boundary != "dirichlet":
            raise ValueError(
                'InfiniteWell.autoestado requiere Grid1D(boundary="dirichlet")'
            )
        if n < 1:
            raise ValueError("n debe ser >= 1 (n=1 es el estado fundamental)")

        L = grid.longitud
        x_rel = grid.x - grid.x_min
        psi = np.sqrt(2.0 / L) * np.sin(n * np.pi * x_rel / L)
        energia = (n**2 * np.pi**2 * hbar**2) / (2 * mass * L**2)
        return WaveFunction(grid, psi.astype(complex)), energia
