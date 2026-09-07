"""Interfaz común para un potencial 2D V(x,y). Análogo a potentials/base.py
(1D) — ver ese docstring para la idea general de autoestado() opcional, que
aquí no se ofrece: ningún potencial 2D de este proyecto es soluble en forma
cerrada todavía.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Potential2D(ABC):
    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Evalúa V(x,y) sobre las mallas X, Y (de Grid2D.meshgrid())."""
