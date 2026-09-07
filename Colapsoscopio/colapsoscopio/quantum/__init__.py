"""Dominio cuántico: TDSE 1D y 2D vía split-operator.

Hermano de `colapsoscopio.classical_waves` (ecuación de onda / Maxwell, FDTD):
comparten la filosofía de separar malla, medio, condición inicial y solver,
pero no comparten código de bajo nivel — son familias de ecuación y de
método numérico distintas (ver el docstring de `classical_waves` para el
porqué). Este paquete existe para que esa asimetría quede explícita en la
estructura de carpetas, no solo en la documentación.
"""

from colapsoscopio.quantum.core.grid import Grid1D
from colapsoscopio.quantum.core.state import WaveFunction
from colapsoscopio.quantum.core.hamiltonian import Hamiltonian1D
from colapsoscopio.quantum.simulation import Simulation, SimulationConfig, Trajectory

from colapsoscopio.quantum.core.grid2d import Grid2D
from colapsoscopio.quantum.core.state2d import WaveFunction2D
from colapsoscopio.quantum.core.hamiltonian2d import Hamiltonian2D
from colapsoscopio.quantum.simulation2d import Simulation2D, SimulationConfig2D, Trajectory2D

__all__ = [
    "Grid1D",
    "WaveFunction",
    "Hamiltonian1D",
    "Simulation",
    "SimulationConfig",
    "Trajectory",
    "Grid2D",
    "WaveFunction2D",
    "Hamiltonian2D",
    "Simulation2D",
    "SimulationConfig2D",
    "Trajectory2D",
]
