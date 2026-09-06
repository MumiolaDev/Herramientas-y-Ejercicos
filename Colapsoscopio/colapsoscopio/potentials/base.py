"""Interfaz común para un potencial V(x).

Un Potential es, en lo esencial, una función V(x) evaluable sobre un array de
numpy. Adicionalmente, los potenciales "solubles" (pozo infinito, oscilador
armónico) saben construir sus propios autoestados analíticos vía
`autoestado(n, grid, hbar, mass)`; eso es lo que permite arrancar una
simulación exactamente en un autoestado y verificar en los tests que la
evolución numérica reproduce la fase e^{-i E_n t / hbar} prevista por teoría,
sin depender de diagonalizar nada numéricamente.

Un potencial nuevo (pozo finito, barrera, doble pozo, Coulomb...) solo
necesita implementar `__call__`; `autoestado` es opcional y por defecto no
está disponible (NotImplementedError), lo cual es correcto: la mayoría de los
potenciales no tienen solución cerrada.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Potential(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evalúa V(x) sobre el array de posiciones x."""

    def autoestado(self, n: int, grid, hbar: float = 1.0, mass: float = 1.0):
        """Construye el autoestado analítico n-ésimo (n=0,1,2,...) como
        WaveFunction, si el potencial es soluble. Devuelve también la energía
        E_n asociada como atributo del resultado para poder comparar contra
        la evolución numérica.
        """
        raise NotImplementedError(
            f"{type(self).__name__} no tiene autoestados analíticos conocidos"
        )
