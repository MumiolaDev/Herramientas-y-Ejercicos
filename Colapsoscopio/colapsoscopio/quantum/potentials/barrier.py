"""Barrera de potencial rectangular: V(x) = V0 para |x - centro| <= ancho/2,
0 fuera. El caso de estudio del efecto túnel: un paquete con energía media
menor que V0 tiene, aun así, probabilidad no nula de aparecer al otro lado.

A diferencia del pozo infinito y el oscilador armónico, esta barrera no
tiene autoestados ligados normalizables con forma cerrada simple (el
problema natural aquí es de scattering —transmisión/reflexión de una onda
que viene de un lado—, no de estados estacionarios en L^2); por eso no
implementa `autoestado()`. La validación para este potencial no es "la
densidad queda estacionaria" sino algo más physical todavía: que aparezca
probabilidad transmitida detrás de la barrera incluso con <E> < V0, y que
la norma y la energía sigan conservándose exactamente igual que en los
otros dos casos (eso no depende de que el potencial sea soluble).
"""

from __future__ import annotations

import numpy as np

from colapsoscopio.quantum.potentials.base import Potential


class PotentialBarrier(Potential):
    def __init__(self, v0: float, centro: float = 0.0, ancho: float = 1.0):
        self.v0 = v0
        self.centro = centro
        self.ancho = ancho

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(np.abs(x - self.centro) <= self.ancho / 2, self.v0, 0.0)

    def transmision_teorica(self, energia: float, hbar: float = 1.0, mass: float = 1.0) -> float:
        """Coeficiente de transmisión T(E) de la barrera rectangular
        *estacionaria* (una onda plana monocromática de energía E, no un
        paquete), fórmula estándar de mecánica cuántica de scattering 1D.
        Sirve solo como referencia aproximada para un paquete: un paquete
        gaussiano no es monocromático, así que su transmisión real es un
        promedio de T(E) pesado por su distribución espectral de energías,
        no T(<E>) evaluado en la energía media.
        """
        a = self.ancho
        if energia <= 0:
            return 0.0
        if energia < self.v0:
            kappa = np.sqrt(2 * mass * (self.v0 - energia)) / hbar
            senh2 = np.sinh(kappa * a) ** 2
            return float(1.0 / (1.0 + (self.v0**2 * senh2) / (4 * energia * (self.v0 - energia))))
        if energia > self.v0:
            k2 = np.sqrt(2 * mass * (energia - self.v0)) / hbar
            sen2 = np.sin(k2 * a) ** 2
            return float(1.0 / (1.0 + (self.v0**2 * sen2) / (4 * energia * (energia - self.v0))))
        # energia == v0: límite especial de la fórmula de arriba
        k = np.sqrt(2 * mass * energia) / hbar
        return float(1.0 / (1.0 + (mass * self.v0 * a**2) / (2 * hbar**2)))
