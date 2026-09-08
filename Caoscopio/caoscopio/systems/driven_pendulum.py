"""Péndulo forzado y amortiguado: θ'' = -(g/l) sin θ - b θ' + A cos(ω_d t).

A diferencia del péndulo simple y el doble (conservativos: la energía
mecánica se conserva, ni se crea ni se destruye), este sistema es
*disipativo* (el término -bθ' quita energía) y *forzado* (A cos(ω_d t) la
repone) — la combinación es la que hace posible algo que un sistema
conservativo no puede tener: un **atractor**. En un péndulo simple o
doble, el volumen del espacio de fases se conserva (teorema de Liouville)
y las trayectorias con distinta condición inicial nunca convergen entre
sí. Acá sí: el volumen se contrae (la disipación lo garantiza), y
trayectorias que arrancan en puntos arbitrariamente distintos terminan,
tras un transiente, sobre el mismo conjunto — un punto fijo, un ciclo
límite, o (para amplitud de forzado suficiente) un **atractor extraño**:
acotado, con estructura fina, pero con sensibilidad exponencial a
condiciones iniciales sobre él, como el péndulo doble.

No es conservativo, así que no implementa energia() (ver
core/trajectory.py: el campo de energía en la Trajectory queda NaN, y
`Simulation.run()` lo detecta solo con intentar llamarlo, sin que este
archivo tenga que declarar nada especial).

Los parámetros por defecto (b=0.5, ω_d=2/3, A variable) son los de un
ejemplo muy citado en la literatura de ecuaciones diferenciales
(θ''+0.5θ'+sinθ=A cos(2t/3)): A=0.9 da un ciclo límite (atractor
periódico, un único punto en la sección de Poincaré estroboscópica);
A=1.15 es caótico (atractor extraño, una nube de puntos con estructura
fina) — verificado numéricamente en este proyecto, no solo citado (ver
tests/test_driven_pendulum.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caoscopio.systems.base import DynamicalSystem


@dataclass
class DrivenDampedPendulum(DynamicalSystem):
    l: float = 1.0
    g: float = 1.0
    b: float = 0.5
    A: float = 1.15
    omega_d: float = 2.0 / 3.0
    dim: int = field(default=2, init=False)

    def derivadas(self, t: float, y: np.ndarray) -> np.ndarray:
        theta, omega = y
        domega = -(self.g / self.l) * np.sin(theta) - self.b * omega + self.A * np.cos(self.omega_d * t)
        return np.array([omega, domega])

    @property
    def periodo_forzado(self) -> float:
        """T_d = 2π/ω_d — el período natural para muestrear una sección de
        Poincaré estroboscópica: tomar el estado una vez por cada período
        de forzado colapsa un ciclo límite de período 1 a un único punto,
        y revela la estructura fina de un atractor extraño."""
        return 2 * np.pi / self.omega_d

    def campo_vectorial(self, theta: np.ndarray, omega: np.ndarray, t: float = 0.0):
        """dθ/dt, dω/dt evaluados en una malla (theta, omega), a fase de
        forzado fija t (por defecto t=0, que es exactamente la fase en la
        que se muestrea la sección de Poincaré estroboscópica —así el
        campo de fondo y los puntos de Poincaré que se dibujan encima
        corresponden al mismo instante del ciclo de forzado—). El sistema
        no es autónomo, así que este es un "congelado" del campo, no un
        campo de flujo verdadero e independiente del tiempo.
        """
        domega = -(self.g / self.l) * np.sin(theta) - self.b * omega + self.A * np.cos(self.omega_d * t)
        return omega, domega
