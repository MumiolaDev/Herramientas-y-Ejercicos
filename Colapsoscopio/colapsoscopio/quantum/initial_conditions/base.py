"""Interfaz común para una condición inicial Psi(x, 0)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class InitialCondition(ABC):
    @abstractmethod
    def construir(self, grid, potential, hbar: float, mass: float):
        """Devuelve la WaveFunction normalizada en t=0.

        Se pasa el `potential` además del `grid` porque algunas condiciones
        iniciales (Eigenstate) necesitan preguntarle al potencial por su
        autoestado analítico; otras (GaussianPacket) lo ignoran.
        """
