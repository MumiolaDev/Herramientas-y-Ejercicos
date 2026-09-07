"""Paquete de onda gaussiano 2D, producto de un gaussiano en x y otro en y
(cada uno de mínima incertidumbre en su propio eje). Análogo a
initial_conditions/gaussian_packet.py (1D) — ver ese docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colapsoscopio.quantum.core.grid2d import Grid2D
from colapsoscopio.quantum.core.state2d import WaveFunction2D


@dataclass
class GaussianPacket2D:
    x0: float
    y0: float
    sigma_x: float
    sigma_y: float
    kx0: float = 0.0
    ky0: float = 0.0

    def construir(self, grid: Grid2D, potential, hbar: float, mass: float) -> WaveFunction2D:
        X, Y = grid.meshgrid()
        envolvente_x = (2 * np.pi * self.sigma_x**2) ** (-0.25) * np.exp(
            -((X - self.x0) ** 2) / (4 * self.sigma_x**2)
        )
        envolvente_y = (2 * np.pi * self.sigma_y**2) ** (-0.25) * np.exp(
            -((Y - self.y0) ** 2) / (4 * self.sigma_y**2)
        )
        fase = np.exp(1j * (self.kx0 * X + self.ky0 * Y))
        psi = envolvente_x * envolvente_y * fase
        return WaveFunction2D(grid, psi).normalizada()
