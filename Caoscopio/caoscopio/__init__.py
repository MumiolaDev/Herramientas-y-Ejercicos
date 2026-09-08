"""Caoscopio: un instrumento para observar la evolución de sistemas dinámicos
en el espacio de fases — caóticos y no caóticos.

Hermano conceptual de `Colapsoscopio` (mismo repo de herramientas, dominio
distinto): allá se integra la ecuación de Schrödinger con un propagador
espectral exacto; acá se integran ecuaciones de movimiento clásicas
(Hamiltonianas, típicamente no separables) con un integrador de propósito
general. La diferencia no es de gusto sino de estructura matemática — ver
el docstring de `core.integrator` para el porqué exacto, y por qué eso
cambia lo que "validar el solver" significa de un proyecto al otro.
"""

from caoscopio.core.trajectory import Simulation, SimulationConfig, Trajectory
from caoscopio.systems.simple_pendulum import SimplePendulum
from caoscopio.systems.double_pendulum import DoublePendulum
from caoscopio.systems.driven_pendulum import DrivenDampedPendulum

__all__ = [
    "Simulation",
    "SimulationConfig",
    "Trajectory",
    "SimplePendulum",
    "DoublePendulum",
    "DrivenDampedPendulum",
]
