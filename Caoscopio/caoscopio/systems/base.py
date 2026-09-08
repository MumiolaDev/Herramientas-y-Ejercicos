"""Interfaz común para un sistema dinámico: un vector de estado y y una regla
dy/dt = f(t, y). No hay más contrato que ese — es deliberadamente mínimo,
para que agregar un sistema nuevo (péndulo forzado, Duffing, Lorenz, tres
cuerpos) sea agregar una clase, no tocar el integrador.

`energia()` es opcional pero, cuando existe, es la validación más
importante que tiene un sistema conservativo: ver core/integrator.py para
por qué acá "se conserva la energía" es una propiedad *medida*, no
*garantizada por construcción* como la norma en Colapsoscopio.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DynamicalSystem(ABC):
    dim: int

    @abstractmethod
    def derivadas(self, t: float, y: np.ndarray) -> np.ndarray:
        """dy/dt en el instante t, estado y. Debe devolver un array de
        forma (self.dim,)."""

    def energia(self, y: np.ndarray) -> float:
        """Energía mecánica total (cinética + potencial) en el estado y,
        si el sistema es conservativo. No todos los sistemas la tienen
        (uno forzado o disipativo, por ejemplo) — de ahí que no sea
        abstracto."""
        raise NotImplementedError(f"{type(self).__name__} no define energia()")
