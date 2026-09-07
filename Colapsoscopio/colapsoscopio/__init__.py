"""Colapsoscopio: un instrumento para observar la evolución temporal de ondas.

El nombre juega con el "colapso" de la función de onda al medir (el problema
de la medición) y el sufijo "-scopio" de todo instrumento de observación
(telescopio, microscopio, ...). Aquí no colapsamos nada de verdad: integramos
una ecuación de onda y *observamos* su evolución en el tiempo.

El proyecto tiene dos dominios físicos, hermanos pero independientes:

- `colapsoscopio.quantum`: la ecuación de Schrödinger dependiente del tiempo
  (1D y 2D), vía el método split-operator. Ver su docstring para el porqué.
- `colapsoscopio.classical_waves`: ecuación de onda clásica / Maxwell (FDTD),
  con su propio solver — no una reutilización del cuántico. Ver su docstring
  para la distinción física entre ambas familias.

Las clases de uso más frecuente del dominio cuántico se re-exportan aquí
para no romper el código existente; todo lo demás se importa explícitamente
desde su submódulo (`colapsoscopio.quantum.potentials`, etc.).
"""

from colapsoscopio.quantum import (
    Grid1D,
    WaveFunction,
    Hamiltonian1D,
    Simulation,
    SimulationConfig,
    Trajectory,
    Grid2D,
    WaveFunction2D,
    Hamiltonian2D,
    Simulation2D,
    SimulationConfig2D,
    Trajectory2D,
)

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
