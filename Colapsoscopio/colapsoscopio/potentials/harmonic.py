"""Oscilador armónico cuántico: V(x) = (1/2) m omega^2 x^2.

Pensado para usarse con Grid1D(..., boundary="periodic") en una caja lo
bastante ancha frente a la longitud característica sqrt(hbar/(m*omega)):
como V no confina exactamente (no hay pared dura), la aproximación de caja
periódica es buena en la medida en que Psi decaiga a ~0 mucho antes de
llegar a los bordes. Un buen punto de partida es tomar
x_max = -x_min ~ 8..10 veces esa longitud característica para los primeros
estados excitados.
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_hermite

from colapsoscopio.potentials.base import Potential


class HarmonicOscillator(Potential):
    def __init__(self, omega: float = 1.0, mass: float = 1.0):
        self.omega = omega
        self.mass = mass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * self.mass * self.omega**2 * x**2

    def autoestado(self, n: int, grid, hbar: float = 1.0, mass: float = 1.0):
        """Autoestado n-ésimo exacto (n=0,1,2,... — n=0 es el estado
        fundamental):

            Psi_n(x) = N_n H_n(y) exp(-y^2/2),   y = sqrt(m*omega/hbar) * x
            E_n = hbar * omega * (n + 1/2)

        N_n se fija numéricamente integrando sobre la malla (en vez de la
        fórmula cerrada 1/sqrt(2^n n! sqrt(pi))*(m*omega/hbar)^{1/4}, que
        para n grande sufre overflow de factorial); eval_hermite es estable
        para n moderado, que es el rango de interés aquí.
        """
        from colapsoscopio.core.state import WaveFunction

        if n < 0:
            raise ValueError("n debe ser >= 0 (n=0 es el estado fundamental)")
        if mass != self.mass:
            raise ValueError(
                "la masa pasada a autoestado() debe coincidir con self.mass del potencial"
            )

        y = np.sqrt(self.mass * self.omega / hbar) * grid.x
        psi_no_normalizada = eval_hermite(n, y) * np.exp(-(y**2) / 2)
        wf = WaveFunction(grid, psi_no_normalizada.astype(complex)).normalizada()
        energia = hbar * self.omega * (n + 0.5)
        return wf, energia
