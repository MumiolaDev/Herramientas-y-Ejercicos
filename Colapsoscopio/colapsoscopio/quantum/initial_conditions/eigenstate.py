"""Condición inicial: arrancar exactamente en el n-ésimo autoestado del
Hamiltoniano (solo disponible para potenciales "solubles", ver
colapsoscopio.quantum.potentials.base.Potential.autoestado).

Es la condición inicial más útil para *validar* un solver: un autoestado
estacionario solo debiera acumular una fase global e^{-i E_n t / hbar}, de
modo que |Psi(x,t)|^2 permanezca exactamente constante en el tiempo. Ver
tests/test_eigenstates.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from colapsoscopio.quantum.initial_conditions.base import InitialCondition


@dataclass
class Eigenstate(InitialCondition):
    n: int
    energia: float = field(default=float("nan"), init=False, repr=True)
    """Se completa tras llamar a construir(); NaN antes de eso."""

    def construir(self, grid, potential, hbar: float, mass: float):
        wf, energia = potential.autoestado(self.n, grid, hbar=hbar, mass=mass)
        self.energia = energia
        return wf
