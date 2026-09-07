"""Estado cuántico Psi(x,y) sobre una Grid2D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colapsoscopio.quantum.core.grid2d import Grid2D


@dataclass
class WaveFunction2D:
    grid: Grid2D
    psi: np.ndarray  # complejo, shape (grid.n_x, grid.n_y)

    def __post_init__(self) -> None:
        self.psi = np.asarray(self.psi, dtype=complex)
        if self.psi.shape != (self.grid.n_x, self.grid.n_y):
            raise ValueError("psi debe tener shape (grid.n_x, grid.n_y)")

    def densidad(self) -> np.ndarray:
        return np.abs(self.psi) ** 2

    def norma(self) -> float:
        return self.grid.integrate(self.densidad())

    def normalizada(self) -> "WaveFunction2D":
        n = self.norma()
        if n <= 0:
            raise ValueError("no se puede normalizar un estado de norma nula")
        return WaveFunction2D(self.grid, self.psi / np.sqrt(n))

    def valor_esperado_x(self) -> float:
        X, _ = self.grid.meshgrid()
        return self.grid.integrate(X * self.densidad())

    def valor_esperado_y(self) -> float:
        _, Y = self.grid.meshgrid()
        return self.grid.integrate(Y * self.densidad())

    def valor_esperado(self, operador_diagonal: np.ndarray) -> float:
        """<A> para un operador diagonal en (x,y), p.ej. el potencial V(x,y)."""
        return self.grid.integrate(operador_diagonal * self.densidad())
