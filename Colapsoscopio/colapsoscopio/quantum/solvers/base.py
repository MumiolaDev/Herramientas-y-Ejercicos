"""Interfaz común para un propagador temporal: avanza Psi un paso dt."""

from __future__ import annotations

from abc import ABC, abstractmethod

from colapsoscopio.quantum.core.state import WaveFunction


class Solver(ABC):
    @abstractmethod
    def paso(self, psi: WaveFunction, dt: float) -> WaveFunction:
        """Devuelve una nueva WaveFunction avanzada en dt (no modifica psi)."""
