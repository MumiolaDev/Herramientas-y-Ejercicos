"""Interfaz común para un potencial 2D V(x,y). Análogo a potentials/base.py
(1D): `autoestado()` es opcional y, cuando existe (como en EmptyBilliard),
sirve para la misma validación fuerte que en 1D — arrancar exactamente en
un autoestado y comprobar que |Psi|^2 queda estacionaria.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Potential2D(ABC):
    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Evalúa V(x,y) sobre las mallas X, Y (de Grid2D.meshgrid())."""

    def autoestado(self, n: int, m: int, grid, hbar: float = 1.0, mass: float = 1.0):
        """Construye el autoestado analítico (n,m), si el potencial es
        soluble. Ver potentials/base.py (1D) para la convención — acá
        también devuelve (WaveFunction2D, energía).
        """
        raise NotImplementedError(
            f"{type(self).__name__} no tiene autoestados analíticos conocidos"
        )
