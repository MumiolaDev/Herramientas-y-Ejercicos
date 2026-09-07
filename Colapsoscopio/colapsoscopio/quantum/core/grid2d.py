"""Malla espacial 2D, con los mismos dos "sabores" de frontera que Grid1D
(ver su docstring para el porqué de cada uno):

- boundary="periodic": FFT2. Pensada para un dominio abierto (doble
  rendija: el paquete nunca debería "sentir" los bordes).
- boundary="dirichlet": DST-I en ambos ejes (`scipy.fft.dstn`/`idstn`,
  type=1), separable porque el operador -∇² con Dirichlet en un rectángulo
  se diagonaliza en el producto tensorial de las bases seno 1D de cada eje:
  sus autovalores de energía cinética son exactamente
  kx_n² + ky_m² = (nπ/Lx)² + (mπ/Ly)², la cuantización real de un **billar
  cuántico rectangular** (Ψ=0 en las cuatro paredes, impuesto por la base
  espectral, no aproximado con paredes "muy altas pero finitas" — mismo
  argumento que el pozo infinito 1D). Un billar con un obstáculo interior
  (billar de Sinai) sigue usando esta misma malla: el obstáculo se agrega
  como potencial alto en su interior, no como condición de frontera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.fft import dstn, idstn

Boundary = Literal["periodic", "dirichlet"]


@dataclass(frozen=True)
class Grid2D:
    x_min: float
    x_max: float
    n_x: int
    y_min: float
    y_max: float
    n_y: int
    boundary: Boundary = "periodic"

    def __post_init__(self) -> None:
        if self.n_x < 8 or self.n_y < 8:
            raise ValueError("n_x y n_y deben ser >= 8 para que el método espectral tenga sentido")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("x_max debe ser mayor que x_min (e igual para y)")
        if self.boundary not in ("periodic", "dirichlet"):
            raise ValueError('boundary debe ser "periodic" o "dirichlet"')

    @property
    def longitud_x(self) -> float:
        return self.x_max - self.x_min

    @property
    def longitud_y(self) -> float:
        return self.y_max - self.y_min

    @property
    def dx(self) -> float:
        if self.boundary == "periodic":
            return self.longitud_x / self.n_x
        return self.longitud_x / (self.n_x + 1)

    @property
    def dy(self) -> float:
        if self.boundary == "periodic":
            return self.longitud_y / self.n_y
        return self.longitud_y / (self.n_y + 1)

    @property
    def x(self) -> np.ndarray:
        """Vector de posiciones en x, shape (n_x,)."""
        if self.boundary == "periodic":
            return self.x_min + self.dx * np.arange(self.n_x)
        return self.x_min + self.dx * np.arange(1, self.n_x + 1)

    @property
    def y(self) -> np.ndarray:
        """Vector de posiciones en y, shape (n_y,)."""
        if self.boundary == "periodic":
            return self.y_min + self.dy * np.arange(self.n_y)
        return self.y_min + self.dy * np.arange(1, self.n_y + 1)

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """X, Y con indexing='ij': X[i,j]=x[i], Y[i,j]=y[j] — así el eje 0 de
        cualquier array 2D de este módulo es x y el eje 1 es y, consistente
        en todo el código (potenciales, estado, densidad)."""
        return np.meshgrid(self.x, self.y, indexing="ij")

    @property
    def kx(self) -> np.ndarray:
        if self.boundary == "periodic":
            return 2 * np.pi * np.fft.fftfreq(self.n_x, d=self.dx)
        n = np.arange(1, self.n_x + 1)
        return n * np.pi / self.longitud_x

    @property
    def ky(self) -> np.ndarray:
        if self.boundary == "periodic":
            return 2 * np.pi * np.fft.fftfreq(self.n_y, d=self.dy)
        n = np.arange(1, self.n_y + 1)
        return n * np.pi / self.longitud_y

    def meshgrid_k(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.kx, self.ky, indexing="ij")

    def transformar_ida(self, psi: np.ndarray) -> np.ndarray:
        if self.boundary == "periodic":
            return np.fft.fft2(psi)
        return dstn(psi.real, type=1, norm="ortho") + 1j * dstn(psi.imag, type=1, norm="ortho")

    def transformar_vuelta(self, psi_hat: np.ndarray) -> np.ndarray:
        if self.boundary == "periodic":
            return np.fft.ifft2(psi_hat)
        return idstn(psi_hat.real, type=1, norm="ortho") + 1j * idstn(psi_hat.imag, type=1, norm="ortho")

    def integrate(self, densidad: np.ndarray) -> float:
        """Integral doble de una densidad real sobre la malla."""
        return float(np.sum(densidad) * self.dx * self.dy)
