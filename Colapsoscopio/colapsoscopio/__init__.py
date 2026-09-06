"""Colapsoscopio: un instrumento para observar la evolución temporal de funciones de onda.

El nombre juega con el "colapso" de la función de onda al medir (el problema
de la medición) y el sufijo "-scopio" de todo instrumento de observación
(telescopio, microscopio, ...). Aquí no colapsamos nada de verdad: integramos
la ecuación de Schrödinger dependiente del tiempo (TDSE) y *observamos*
|Psi(x,t)|^2, Re/Im Psi, fase, valores esperados, etc.
"""

from colapsoscopio.core.grid import Grid1D
from colapsoscopio.core.state import WaveFunction
from colapsoscopio.core.hamiltonian import Hamiltonian1D
from colapsoscopio.simulation import Simulation, SimulationConfig, Trajectory

__all__ = [
    "Grid1D",
    "WaveFunction",
    "Hamiltonian1D",
    "Simulation",
    "SimulationConfig",
    "Trajectory",
]
