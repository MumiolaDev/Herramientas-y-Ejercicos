"""Ecuación de estado de Carnahan-Starling: la termodinámica del agua de
este autómata.

    p(ρ, T) = ρRT (1 + x + x² − x³)/(1 − x)³ − aρ²,    x = bρ/4

El primer término es la esfera dura de Carnahan-Starling (una resumación
excelente de la serie del virial para esferas duras); el segundo, la
atracción de campo medio de van der Waals. Por debajo de T_c la isoterma
tiene el lazo de van der Waals y hay coexistencia líquido-vapor. En
unidades de red se usa a = 1, b = 4, R = 1 (Yuan & Schaefer 2006), lo que
da T_c ≈ 0.0943, ρ_c ≈ 0.1304.

Cómo entra en lattice Boltzmann (Shan & Chen 1993; Yuan & Schaefer 2006):
la parte ideal c_s²ρ la pone la red gratis; el resto se inyecta como una
fuerza entre vecinos derivada de un "pseudopotencial" ψ elegido tal que

    p_EOS = c_s² ρ + (G c_s²/2) ψ²   ⇒   ψ = sqrt( 2(p_EOS − c_s² ρ) / (G c_s²) ),  G = −1.

Esa fuerza F = −G ψ(x) Σ w_i ψ(x+e_i) e_i es LOCAL (sólo vecinos) — por eso
todo el modelo sigue siendo un autómata celular — y en el continuo se
expande como −∇(p_EOS − c_s²ρ) más términos de gradiente que generan la
tensión superficial. La interfaz líquido-vapor no se rastrea: emerge.

Energía interna: para cualquier EOS de la forma p = T·φ(ρ) − aρ², la
identidad termodinámica (∂u/∂v)_T = T(∂p/∂T)_v − p da (∂u/∂v)_T = aρ², o
sea u = c_v T − aρ por unidad de masa. La parte de esferas duras es
puramente entrópica: no aporta energía. La energía de cohesión −aρ² por
unidad de volumen es de donde sale el calor latente.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

from aguacero.lattice import CS2

G_PSEUDOPOTENCIAL = -1.0


@dataclass(frozen=True)
class CarnahanStarling:
    a: float = 1.0
    b: float = 4.0
    R: float = 1.0
    cv: float = 6.0  # calor específico a volumen constante (Li et al. 2015 usan este valor)

    @property
    def T_critica(self) -> float:
        return 0.3773 * self.a / (self.b * self.R)

    @property
    def rho_critica(self) -> float:
        return 0.5218 / self.b

    def _factor_esfera_dura(self, rho):
        x = self.b * rho / 4.0
        return (1.0 + x + x * x - x**3) / (1.0 - x) ** 3

    def presion(self, rho, T):
        return rho * self.R * T * self._factor_esfera_dura(rho) - self.a * rho * rho

    def dp_dT(self, rho, T=None):
        """(∂p/∂T)_ρ — el coeficiente del término de trabajo de compresión
        en la ecuación de energía (ver aguacero/termico.py)."""
        return rho * self.R * self._factor_esfera_dura(rho)

    def psi(self, rho, T):
        exceso = self.presion(rho, T) - CS2 * rho
        # con G < 0 el radicando es −2·exceso/c_s²; es positivo donde la
        # EOS está por debajo del gas ideal de la red, que es todo el
        # rango de interés (ρ ≲ 0.5). El máximo con 0 sólo protege de
        # densidades absurdas durante un transiente violento.
        return np.sqrt(np.maximum(2.0 * exceso / (G_PSEUDOPOTENCIAL * CS2), 0.0))

    def densidad_energia_interna(self, rho, T):
        return rho * self.cv * T - self.a * rho * rho

    def coexistencia(self, T: float, epsilon: float | None = 0.0) -> tuple[float, float]:
        """Densidades (ρ_vapor, ρ_líquido) en equilibrio a temperatura T.

        epsilon=None → construcción de Maxwell (igualdad de áreas en
        ∫ (p0 − p) dρ/ρ²): la termodinámica "de verdad" de esta EOS.

        epsilon=ε → la condición de equilibrio MECÁNICO del modelo de
        pseudopotencial (Shan 2008; Li, Luo & Li 2012):

            ∫_{ρg}^{ρl} (p0 − p_EOS) ψ'/ψ^{1+ε} dρ = 0.

        Con el forzamiento de Guo que usa este simulador ε = 0, y NO
        coincide con Maxwell: es la famosa inconsistencia termodinámica del
        pseudopotencial. El simulador reproduce la predicción ε = 0 a 4-5
        cifras (tests/test_coexistencia.py), así que lo que se desvía de
        Maxwell es el modelo discreto, no un error de implementación.
        """
        rho_max_p = minimize_scalar(lambda r: -self.presion(r, T), bounds=(1e-4, 0.25), method="bounded").x
        rho_min_p = minimize_scalar(lambda r: self.presion(r, T), bounds=(rho_max_p, 0.7), method="bounded").x
        p_alto = self.presion(rho_max_p, T)
        p_bajo = max(self.presion(rho_min_p, T), self.presion(1e-7, T))

        def psi_escalar(r):
            return float(self.psi(np.asarray(r), T))

        def dpsi(r, h=1e-7):
            return (psi_escalar(r + h) - psi_escalar(r - h)) / (2 * h)

        def raices(p0):
            rg = brentq(lambda r: self.presion(r, T) - p0, 1e-8, rho_max_p)
            rl = brentq(lambda r: self.presion(r, T) - p0, rho_min_p, 0.95 * 4 / self.b)
            return rg, rl

        def condicion(p0):
            rg, rl = raices(p0)
            if epsilon is None:
                integrando = lambda r: (p0 - self.presion(r, T)) / r**2
            else:
                integrando = lambda r: (p0 - self.presion(r, T)) * dpsi(r) / psi_escalar(r) ** (1 + epsilon)
            return quad(integrando, rg, rl, limit=400, points=[rho_max_p, rho_min_p])[0]

        a_, b_ = p_bajo * (1 + 1e-9), p_alto * (1 - 1e-9)
        if condicion(a_) * condicion(b_) > 0:
            raise ValueError(
                f"sin coexistencia para ε={epsilon} a T/T_c={T / self.T_critica:.3f}: con ε pequeño la "
                "densidad de vapor de este esquema tiende a 0 al enfriar (usa sigma_li > 0 o una T mayor)"
            )
        p0 = brentq(condicion, a_, b_)
        return raices(p0)
