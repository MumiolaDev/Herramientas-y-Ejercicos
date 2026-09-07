"""Paquete de onda gaussiano de mínima incertidumbre (Delta x * Delta p = hbar/2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colapsoscopio.quantum.initial_conditions.base import InitialCondition


@dataclass
class GaussianPacket(InitialCondition):
    x0: float
    """Posición donde está centrado el paquete."""

    sigma: float
    """Ancho (desviación estándar de |Psi|^2, no de Psi)."""

    k0: float = 0.0
    """Número de onda medio: <p> inicial = hbar * k0. Positivo = se mueve
    hacia +x, negativo hacia -x, cero = en reposo (dispersión pura)."""

    def construir(self, grid, potential, hbar: float, mass: float):
        from colapsoscopio.quantum.core.state import WaveFunction

        envolvente = (2 * np.pi * self.sigma**2) ** (-0.25) * np.exp(
            -((grid.x - self.x0) ** 2) / (4 * self.sigma**2)
        )
        fase = np.exp(1j * self.k0 * grid.x)
        psi = envolvente * fase
        return WaveFunction(grid, psi).normalizada()
