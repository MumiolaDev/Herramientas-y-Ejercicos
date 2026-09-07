"""Malla espacial 2D, periódica en ambos ejes.

Mismo principio que Grid1D (ver su docstring), pero solo con boundary
periódica: no hay un análogo simple de la transformada seno 2D para un
confinamiento rectangular arbitrario que valga la pena antes de tener un
caso de uso real que lo pida (un "billar cuántico" rectangular sí lo tendría
—DST en ambos ejes—, pero no es parte de este alcance). Para 2D, con FFT2 en
una caja lo bastante ancha alcanza para doble rendija y billares "blandos".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid2D:
    x_min: float
    x_max: float
    n_x: int
    y_min: float
    y_max: float
    n_y: int

    def __post_init__(self) -> None:
        if self.n_x < 8 or self.n_y < 8:
            raise ValueError("n_x y n_y deben ser >= 8 para que la FFT2 tenga sentido")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("x_max debe ser mayor que x_min (e igual para y)")

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.n_x

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.n_y

    @property
    def x(self) -> np.ndarray:
        """Vector de posiciones en x, shape (n_x,)."""
        return self.x_min + self.dx * np.arange(self.n_x)

    @property
    def y(self) -> np.ndarray:
        """Vector de posiciones en y, shape (n_y,)."""
        return self.y_min + self.dy * np.arange(self.n_y)

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """X, Y con indexing='ij': X[i,j]=x[i], Y[i,j]=y[j] — así el eje 0 de
        cualquier array 2D de este módulo es x y el eje 1 es y, consistente
        en todo el código (potenciales, estado, densidad)."""
        return np.meshgrid(self.x, self.y, indexing="ij")

    @property
    def kx(self) -> np.ndarray:
        return 2 * np.pi * np.fft.fftfreq(self.n_x, d=self.dx)

    @property
    def ky(self) -> np.ndarray:
        return 2 * np.pi * np.fft.fftfreq(self.n_y, d=self.dy)

    def meshgrid_k(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.kx, self.ky, indexing="ij")

    def transformar_ida(self, psi: np.ndarray) -> np.ndarray:
        return np.fft.fft2(psi)

    def transformar_vuelta(self, psi_hat: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(psi_hat)

    def integrate(self, densidad: np.ndarray) -> float:
        """Integral doble de una densidad real sobre la malla."""
        return float(np.sum(densidad) * self.dx * self.dy)
