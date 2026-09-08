"""Péndulo simple: el caso de referencia *no caótico* (1 grado de libertad,
integrable — su retrato de fases es siempre una curva cerrada, nunca se
cruza consigo misma). Sirve como el sistema "de control" para contrastar
contra el péndulo doble, y como el caso de validación fuerte: para
amplitud pequeña, el período numérico debe converger al período de
oscilador armónico T=2π√(l/g) (ver tests/test_pendulo_simple.py).

Estado y = [θ, ω] (ángulo desde la vertical, velocidad angular).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caoscopio.systems.base import DynamicalSystem


@dataclass
class SimplePendulum(DynamicalSystem):
    l: float = 1.0
    g: float = 9.81
    dim: int = field(default=2, init=False)

    def derivadas(self, t: float, y: np.ndarray) -> np.ndarray:
        theta, omega = y
        return np.array([omega, -(self.g / self.l) * np.sin(theta)])

    def energia(self, y: np.ndarray) -> float:
        theta, omega = y
        cinetica = 0.5 * self.l**2 * omega**2
        potencial = -self.g * self.l * np.cos(theta)
        return cinetica + potencial

    def posicion(self, y: np.ndarray) -> tuple[float, float]:
        """(x, y) de la masa, con el pivote en el origen y la vertical
        hacia abajo en -y."""
        theta = y[0]
        return (self.l * np.sin(theta), -self.l * np.cos(theta))
