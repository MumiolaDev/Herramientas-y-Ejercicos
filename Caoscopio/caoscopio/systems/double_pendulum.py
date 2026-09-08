"""Péndulo doble: el ejemplo de libro de caos determinista en un sistema
mecánico simple — dos grados de libertad, Hamiltoniano no separable
(la energía cinética mezcla θ1' y θ2' con un coseno de la diferencia de
ángulos), suficiente para movimiento caótico a energías altas y
quasi-periódico (regular) a energías bajas. Estado y = [θ1, ω1, θ2, ω2].

Las ecuaciones de movimiento (vía Lagrange) son las estándar de la
literatura; se transcriben tal cual, no se re-derivan acá. La validación
de que están bien transcritas —y no solo "autoconsistentes"— es el punto
importante: conservar energía por sí solo NO lo garantiza, porque una
energia() con el mismo error que derivadas() seguiría "conservándose".
La prueba independiente es el límite m2→0: con masa nula en el segundo
péndulo, este no puede ejercer torque sobre el primero, así que θ1(t)
debe reducirse exactamente a un péndulo simple de longitud l1 — y en
efecto, tomando ese límite en las fórmulas de abajo, el término de
acoplamiento se cancela y dω1/dt → -(g/l1) sin θ1. Ver
tests/test_double_pendulum.py, que corre ambos sistemas con la misma
condición inicial en θ1 y compara las trayectorias.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caoscopio.systems.base import DynamicalSystem


@dataclass
class DoublePendulum(DynamicalSystem):
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81
    dim: int = field(default=4, init=False)

    def derivadas(self, t: float, y: np.ndarray) -> np.ndarray:
        th1, w1, th2, w2 = y
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        delta = th1 - th2
        sin_d, cos_d = np.sin(delta), np.cos(delta)
        den = 2 * m1 + m2 - m2 * np.cos(2 * delta)

        dw1 = (
            -g * (2 * m1 + m2) * np.sin(th1)
            - m2 * g * np.sin(th1 - 2 * th2)
            - 2 * sin_d * m2 * (w2**2 * l2 + w1**2 * l1 * cos_d)
        ) / (l1 * den)

        dw2 = (
            2
            * sin_d
            * (w1**2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(th1) + w2**2 * l2 * m2 * cos_d)
        ) / (l2 * den)

        return np.array([w1, dw1, w2, dw2])

    def energia(self, y: np.ndarray) -> float:
        th1, w1, th2, w2 = y
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        cinetica = (
            0.5 * (m1 + m2) * l1**2 * w1**2
            + 0.5 * m2 * l2**2 * w2**2
            + m2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2)
        )
        potencial = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
        return cinetica + potencial

    def posiciones(self, y: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
        """((x1,y1), (x2,y2)) de las dos masas, pivote en el origen,
        vertical hacia abajo en -y."""
        th1, _, th2, _ = y
        x1, y1 = self.l1 * np.sin(th1), -self.l1 * np.cos(th1)
        x2, y2 = x1 + self.l2 * np.sin(th2), y1 - self.l2 * np.cos(th2)
        return (x1, y1), (x2, y2)
