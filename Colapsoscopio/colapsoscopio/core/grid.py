"""Discretización espacial 1D, con dos "sabores" de condición de frontera.

- boundary="periodic": malla estándar para el método split-step de Fourier.
  Pensada para estados ligados en una caja "abierta" lo bastante ancha frente
  a la extensión de Psi (p.ej. el oscilador armónico): la condición periódica
  es una conveniencia numérica, no una propiedad física del sistema, así que
  hay que elegir la caja con margen para que el error de aliasing en los
  bordes sea despreciable.

- boundary="dirichlet": malla de N puntos *interiores* estrictamente entre
  x_min y x_max, con Psi = 0 exactamente en ambos bordes (que no forman parte
  de la malla). Esta es la condición de frontera correcta para un pozo de
  potencial infinito: el confinamiento no se modela como una función V(x)
  gigante, sino como la condición de borde misma. Su base espectral natural
  es la transformada seno discreta (DST-I), cuyos autovalores de energía
  cinética son exactamente k_n = n*pi/L, n=1..N — la misma cuantización que
  predice la teoría para el pozo infinito.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.fft import dst, idst

Boundary = Literal["periodic", "dirichlet"]


@dataclass(frozen=True)
class Grid1D:
    x_min: float
    x_max: float
    n_points: int
    boundary: Boundary = "periodic"

    def __post_init__(self) -> None:
        if self.n_points < 8:
            raise ValueError("n_points debe ser >= 8 para que el método espectral tenga sentido")
        if self.x_max <= self.x_min:
            raise ValueError("x_max debe ser mayor que x_min")
        if self.boundary not in ("periodic", "dirichlet"):
            raise ValueError('boundary debe ser "periodic" o "dirichlet"')

    @property
    def longitud(self) -> float:
        return self.x_max - self.x_min

    @property
    def dx(self) -> float:
        if self.boundary == "periodic":
            return self.longitud / self.n_points
        # dirichlet: n_points nodos interiores, más los dos bordes fijos en 0
        return self.longitud / (self.n_points + 1)

    @property
    def x(self) -> np.ndarray:
        if self.boundary == "periodic":
            return self.x_min + self.dx * np.arange(self.n_points)
        return self.x_min + self.dx * np.arange(1, self.n_points + 1)

    @property
    def k(self) -> np.ndarray:
        """Números de onda de la base espectral en que T = hbar^2 k^2 / 2m
        es diagonal: frecuencias de Fourier si es periódica, o k_n = n*pi/L
        (autovalores de -d^2/dx^2 con Dirichlet) si es dirichlet.
        """
        if self.boundary == "periodic":
            return 2 * np.pi * np.fft.fftfreq(self.n_points, d=self.dx)
        n = np.arange(1, self.n_points + 1)
        return n * np.pi / self.longitud

    def transformar_ida(self, psi: np.ndarray) -> np.ndarray:
        """De representación de posición a la base espectral donde T es diagonal."""
        if self.boundary == "periodic":
            return np.fft.fft(psi)
        return dst(psi.real, type=1, norm="ortho") + 1j * dst(psi.imag, type=1, norm="ortho")

    def transformar_vuelta(self, psi_hat: np.ndarray) -> np.ndarray:
        """Inversa de transformar_ida."""
        if self.boundary == "periodic":
            return np.fft.ifft(psi_hat)
        return idst(psi_hat.real, type=1, norm="ortho") + 1j * idst(psi_hat.imag, type=1, norm="ortho")

    def integrate(self, densidad: np.ndarray) -> float:
        """Integral de una densidad real sobre la malla (regla del rectángulo)."""
        return float(np.sum(densidad) * self.dx)
