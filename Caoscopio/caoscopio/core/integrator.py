"""Runge-Kutta de 4to orden, de propósito general.

Por qué esto es distinto —en un sentido estructural, no solo de gusto—
del propagador de Colapsoscopio: el split-operator explota que
H = T + V se separa en dos piezas, cada una diagonal en una base distinta
(posición para V, momento para T), así que e^{-iHdt/ℏ} se factoriza en
exponenciales exactas que son, cada una, una simple multiplicación por una
fase — de ahí la unitariedad a precisión de máquina. El Hamiltoniano de un
péndulo doble no se separa así: la energía cinética mezcla θ1' y θ2' con
un factor cos(θ1-θ2) que depende de la propia posición, así que no existe
una base fija donde "la parte cinética" sea diagonal de una vez y para
siempre. Sin esa estructura, no hay integrador con conservación de energía
*exacta* que sea, a la vez, simple y de propósito general — de ahí el RK4:
un método estándar, preciso (error local O(dt^5), global O(dt^4)), pero
sin ninguna garantía algebraica de conservación. La energía se conserva
*aproximadamente*, con una deriva que depende de dt y se mide, no se
declara — exactamente lo que validan los tests de este proyecto.

(Los métodos que sí garantizan algo estructural para sistemas Hamiltonianos
—los integradores simplécticos, como Störmer-Verlet/leapfrog— existen y
conservan una energía *sombra* cercana a la real sin deriva secular, pero
requieren separar el Hamiltoniano en cinética+potencial desacopladas, que
el péndulo doble tampoco cumple sin una transformación de coordenadas
adicional. Queda en el roadmap si la deriva de RK4 llega a ser un problema
real para alguna trayectoria de interés.)
"""

from __future__ import annotations

import numpy as np

from caoscopio.systems.base import DynamicalSystem


class RK4Integrator:
    def __init__(self, sistema: DynamicalSystem):
        self.sistema = sistema

    def paso(self, t: float, y: np.ndarray, dt: float) -> np.ndarray:
        f = self.sistema.derivadas
        k1 = f(t, y)
        k2 = f(t + dt / 2, y + dt / 2 * k1)
        k3 = f(t + dt / 2, y + dt / 2 * k2)
        k4 = f(t + dt, y + dt * k3)
        return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
