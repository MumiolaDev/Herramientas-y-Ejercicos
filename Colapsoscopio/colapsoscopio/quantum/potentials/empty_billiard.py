"""Billar rectangular vacío: V(x,y) = 0 en todo el interior. El confinamiento
no lo da esta clase sino Grid2D(..., boundary="dirichlet") —igual que
InfiniteWell en 1D—: úsense siempre juntos.

Es el único potencial 2D de este proyecto con autoestados analíticos, porque
el problema es separable: Psi_{n,m}(x,y) = psi_n(x) * psi_m(y), producto de
dos autoestados 1D de pozo infinito, uno por eje. Sirve como el caso de
validación fuerte del DST-2D —arrancar ahí y comprobar que |Psi|^2 queda
exactamente estacionaria— igual que InfiniteWell.autoestado() en 1D.
"""

from __future__ import annotations

import numpy as np

from colapsoscopio.quantum.potentials.base2d import Potential2D


class EmptyBilliard(Potential2D):
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.zeros_like(X)

    def autoestado(self, n: int, m: int, grid, hbar: float = 1.0, mass: float = 1.0):
        from colapsoscopio.quantum.core.state2d import WaveFunction2D

        if grid.boundary != "dirichlet":
            raise ValueError(
                'EmptyBilliard.autoestado requiere Grid2D(boundary="dirichlet")'
            )
        if n < 1 or m < 1:
            raise ValueError("n y m deben ser >= 1 (n=m=1 es el estado fundamental)")

        Lx, Ly = grid.longitud_x, grid.longitud_y
        X, Y = grid.meshgrid()
        x_rel, y_rel = X - grid.x_min, Y - grid.y_min
        psi = (
            np.sqrt(4.0 / (Lx * Ly))
            * np.sin(n * np.pi * x_rel / Lx)
            * np.sin(m * np.pi * y_rel / Ly)
        )
        energia = (np.pi**2 * hbar**2 / (2 * mass)) * ((n / Lx) ** 2 + (m / Ly) ** 2)
        return WaveFunction2D(grid, psi.astype(complex)), energia
