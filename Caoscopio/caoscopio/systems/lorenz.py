"""El atractor de Lorenz: ẋ=σ(y-x), ẏ=x(ρ-z)-y, ż=xy-βz.

El primer sistema del proyecto que no es un péndulo —no hay ángulo, no hay
gravedad— y el primero en 3D. Lorenz lo derivó en 1963 como una
simplificación drástica (3 modos) de la convección atmosférica de
Rayleigh-Bénard; con los parámetros clásicos (σ=10, ρ=28, β=8/3) es
caótico y disipativo, y su atractor extraño —la "mariposa"— es la imagen
más reconocible de la teoría del caos.

Dos validaciones independientes de que las ecuaciones están bien, ninguna
apoyada en "converge a donde yo digo que converge":

1. **Puntos fijos exactos**: para ρ>1 el sistema tiene tres equilibrios
   —el origen y el par simétrico C± = (±√(β(ρ-1)), ±√(β(ρ-1)), ρ-1)—,
   cerrados en forma analítica. Evaluar `derivadas()` en cualquiera de los
   tres debe dar cero: es una prueba algebraica directa, no una
   observación de "parece que converge ahí".
2. **Contracción de volumen a la tasa exacta**: la divergencia del campo
   vectorial, ∇·f = -σ-1-β, es *constante* (no depende de dónde se evalúe)
   — así que cualquier volumen infinitesimal en el espacio de fases se
   contrae exactamente como e^{(-σ-1-β)t}, sin importar su forma ni dónde
   esté. Se verifica tomando un tetraedro de cuatro puntos infinitesimalmente
   cercanos, integrando los cuatro, y comparando la tasa de decaimiento de
   su volumen contra -σ-1-β: coinciden a 0.1% (ver
   tests/test_lorenz.py). Es la misma idea que "energía conservada" en un
   sistema Hamiltoniano, pero para un sistema disipativo: no es que algo
   se conserve, es que se *contrae a una tasa que se puede predecir de
   antemano* — y eso es justamente lo que hace posible que exista un
   atractor de todos modos: el sistema es caótico (las trayectorias
   dentro del atractor divergen exponencialmente entre sí) y a la vez
   disipativo (el volumen que las contiene se colapsa) — el atractor
   mismo es un conjunto de volumen cero pero de "longitud" infinita, la
   definición informal de fractal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caoscopio.systems.base import DynamicalSystem


@dataclass
class Lorenz(DynamicalSystem):
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dim: int = field(default=3, init=False)

    def derivadas(self, t: float, y: np.ndarray) -> np.ndarray:
        x, y_, z = y
        return np.array([
            self.sigma * (y_ - x),
            x * (self.rho - z) - y_,
            x * y_ - self.beta * z,
        ])

    @property
    def divergencia(self) -> float:
        """∇·f, constante en todo el espacio de fases (no depende de
        x,y,z) — ver el docstring del módulo."""
        return -self.sigma - 1 - self.beta

    def puntos_fijos(self) -> list[np.ndarray]:
        """Los equilibrios exactos del sistema, válidos para ρ>1."""
        if self.rho <= 1:
            return [np.zeros(3)]
        c = np.sqrt(self.beta * (self.rho - 1))
        return [
            np.zeros(3),
            np.array([c, c, self.rho - 1]),
            np.array([-c, -c, self.rho - 1]),
        ]
