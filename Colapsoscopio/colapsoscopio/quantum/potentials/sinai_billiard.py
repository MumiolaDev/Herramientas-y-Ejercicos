"""Billar de Sinai: el billar rectangular vacío (ver empty_billiard.py) con
un obstáculo circular removido del centro. Es el ejemplo canónico de
billar caóticamente disperso en la literatura de caos clásico y cuántico
—la trayectoria clásica que rebota contra el disco central es sensible a
condiciones iniciales incluso sin ninguna otra irregularidad en la
geometría—, en contraste con el billar rectangular vacío, que es
integrable (separable en x,y).

El confinamiento exterior lo sigue dando Grid2D(..., boundary="dirichlet")
—no esta clase—; esta clase solo agrega el disco central como una región
de potencial alto, exactamente como PotentialBarrier agrega una barrera
sobre un dominio ya confinado en 1D.
"""

from __future__ import annotations

import numpy as np

from colapsoscopio.quantum.potentials.base2d import Potential2D


class SinaiBilliard(Potential2D):
    def __init__(self, v0_obstaculo: float, centro: tuple[float, float] = (0.0, 0.0), radio: float = 1.0):
        self.v0_obstaculo = v0_obstaculo
        self.centro = centro
        self.radio = radio

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        cx, cy = self.centro
        dentro_del_disco = (X - cx) ** 2 + (Y - cy) ** 2 <= self.radio**2
        return np.where(dentro_del_disco, self.v0_obstaculo, 0.0)
