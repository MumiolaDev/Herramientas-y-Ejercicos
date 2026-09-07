"""Doble rendija: una pantalla delgada (V=v0, alto) en x=x_pantalla, opaca
salvo por dos aberturas verticales (V=0) centradas en y=+separacion/2 y
y=-separacion/2, cada una de ancho `ancho_rendija`. Un paquete que incide
desde la izquierda con momento en +x se difracta al pasar, y el patrón de
interferencia aparece en la densidad |Psi|^2 detrás de la pantalla.
"""

from __future__ import annotations

import numpy as np

from colapsoscopio.quantum.potentials.base2d import Potential2D


class DoubleSlit(Potential2D):
    def __init__(
        self,
        v0: float,
        x_pantalla: float = 0.0,
        grosor: float = 0.5,
        separacion: float = 3.0,
        ancho_rendija: float = 0.8,
    ):
        self.v0 = v0
        self.x_pantalla = x_pantalla
        self.grosor = grosor
        self.separacion = separacion
        self.ancho_rendija = ancho_rendija

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        en_pantalla = np.abs(X - self.x_pantalla) <= self.grosor / 2
        centro_1 = self.separacion / 2
        centro_2 = -self.separacion / 2
        en_rendija_1 = np.abs(Y - centro_1) <= self.ancho_rendija / 2
        en_rendija_2 = np.abs(Y - centro_2) <= self.ancho_rendija / 2
        en_rendija = en_rendija_1 | en_rendija_2
        return np.where(en_pantalla & ~en_rendija, self.v0, 0.0)
