"""El autómata completo: una grilla de pixeles, cada uno fluido, pared o
fuente térmica, que evoluciona con reglas locales.

Un paso de tiempo, en orden:

  1. momentos     ρ = Σ f_i,  ρu = Σ f_i e_i + F/2       (u "física" de Guo)
  2. fuerza       F = F_Shan-Chen(ψ(ρ,T)) + ρ g ŷ          (cohesión + gravedad)
  3. colisión     f_i ← f_i − (f_i − f_i^eq)/τ + S_i(F) + C_i   (BGK + Guo et al. 2002
                  + corrección de consistencia termodinámica de Li, Luo & Li 2013)
  4. propagación  f_i(x + e_i) ← f_i(x)
  5. paredes      rebote completo: en celdas sólidas f_i ↔ f_opuesto(i)
  6. temperatura  un paso de la ecuación de energía (aguacero/termico.py)

No hay "reglas de caída" como en un falling-sand: el agua cae porque la
gravedad es una fuerza en la ecuación de momento, forma gotas porque la
EOS tiene un lazo de van der Waals, y hierve porque la temperatura entra
en esa misma EOS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from aguacero.eos import G_PSEUDOPOTENCIAL, CarnahanStarling
from aguacero.lattice import E, EX, EY, OPUESTO, W, W3, desplazar, equilibrio, propagar
from aguacero.termico import paso_temperatura


@dataclass
class Parametros:
    T_reducida: float = 0.7  # temperatura inicial y de referencia, en unidades de T_c
    tau: float = 1.0  # tiempo de relajación BGK → viscosidad ν = (τ − 1/2)/3
    gravedad: float = 5e-5  # unidades de red; caída de 100 px → |u| ≈ √(2gh) = 0.1, aún bajo Mach (ver README: número de Bond)
    chi: float = 0.08  # difusividad térmica
    mojabilidad: float = 0.15  # 0 = la pared "ve" vapor (hidrofóbica); estable hasta ~0.25, ver README
    sigma_li: float = 0.33  # corrección de consistencia termodinámica (ver _correccion_li); 0 = Guo puro
    T_min_reducida: float = 0.55  # ver _psi_seguro
    T_max_reducida: float = 1.6
    isotermico: bool = False  # True congela T (no se integra la ecuación de energía)
    eos: CarnahanStarling = field(default_factory=CarnahanStarling)

    @property
    def viscosidad(self) -> float:
        return (self.tau - 0.5) / 3.0


def _resolver_motor(motor: str) -> str:
    if motor not in ("auto", "numpy", "numba"):
        raise ValueError(f"motor desconocido: {motor!r}")
    if motor == "numpy":
        return "numpy"
    try:
        import numba  # noqa: F401
    except ImportError:
        if motor == "numba":
            raise
        return "numpy"
    return "numba"


# κ en ε = κσ, medido (ver Mundo._correccion_li y tests/test_coexistencia.py)
KAPPA_LI = 5.2
FORMA_LI = (3.0 * W * (3.0 * (E**2).sum(axis=1) - 2.0))[:, None, None]


class Mundo:
    def __init__(
        self,
        ancho: int,
        alto: int,
        parametros: Parametros | None = None,
        marco: bool = True,
        motor: str = "auto",
    ):
        """motor: "numpy" (referencia legible), "numba" (el mismo paso
        compilado, ~15× más rápido) o "auto" (numba si está instalado)."""
        self.p = parametros or Parametros()
        self.motor = _resolver_motor(motor)
        self.ancho, self.alto = ancho, alto
        eos = self.p.eos
        self.T0 = self.p.T_reducida * eos.T_critica
        self.epsilon = KAPPA_LI * self.p.sigma_li
        self.rho_vapor, self.rho_liquido = eos.coexistencia(self.T0, epsilon=self.epsilon)

        self.solido = np.zeros((alto, ancho), dtype=bool)
        if marco:
            self.solido[0, :] = self.solido[-1, :] = True
            self.solido[:, 0] = self.solido[:, -1] = True
        self.fuente = np.zeros_like(self.solido)
        self.T_fuente = np.full((alto, ancho), self.T0)

        cero = np.zeros((alto, ancho))
        self.f = equilibrio(np.full((alto, ancho), self.rho_vapor), cero, cero)
        self.T = np.full((alto, ancho), self.T0)
        self.rho = self.f.sum(axis=0)
        self.ux = cero.copy()
        self.uy = cero.copy()
        self.pasos = 0
        self.masa_agregada = 0.0  # por las herramientas (grifo, borrar…): el paso en sí no crea masa

        self.altura = (alto - 1 - np.arange(alto))[:, None] * np.ones((1, ancho))

    # ------------------------------------------------------------------ física

    @property
    def fluido(self) -> np.ndarray:
        return ~self.solido

    def _psi_seguro(self, rho, T):
        """Muy por debajo de T_c la densidad de vapor en coexistencia se
        vuelve diminuta y BGK se desestabiliza. Se recorta T SÓLO al evaluar
        ψ (la temperatura en sí no se toca), como red de seguridad: con la
        corrección de Li activa el rango estable llega a ~0.6 T_c y el
        recorte, en T_min_reducida, no actúa en las escenas incluidas."""
        Tc = self.p.eos.T_critica
        T_psi = np.clip(T, self.p.T_min_reducida * Tc, self.p.T_max_reducida * Tc)
        return self.p.eos.psi(rho, T_psi)

    def _fuerza_cohesion(self, rho):
        """F = −G ψ(x) Σ w_i ψ(x+e_i) e_i. Devuelve también ψ (la usa la
        corrección de Li)."""
        fluido = self.fluido
        psi = self._psi_seguro(rho, self.T)
        rho_pared = self.rho_vapor + self.p.mojabilidad * (self.rho_liquido - self.rho_vapor)
        psi_pared = float(self.p.eos.psi(np.asarray(rho_pared), self.T0))
        psi = np.where(fluido, psi, psi_pared)

        sx = np.zeros_like(rho)
        sy = np.zeros_like(rho)
        for i in range(1, 9):
            vecino = desplazar(psi, i)
            sx += W[i] * vecino * E[i, 0]
            sy += W[i] * vecino * E[i, 1]
        Fx = -G_PSEUDOPOTENCIAL * psi * sx
        Fy = -G_PSEUDOPOTENCIAL * psi * sy
        return np.where(fluido, Fx, 0.0), np.where(fluido, Fy, 0.0), psi

    def _correccion_li(self, Fx, Fy, psi):
        """Término de Li, Luo & Li (2013), trasladado de MRT a BGK.

        El pseudopotencial con forzamiento de Guo es mecánicamente estable
        pero no termodinámicamente consistente: su coexistencia obedece
        ∫(p0 − p)ψ'/ψ^{1+ε} dρ = 0 con ε = 0, no la construcción de Maxwell.
        Li et al. agregan una fuente que sólo toca los momentos de energía
        e y ε de D2Q9 (δe = +12Q, δε = −12Q, masa, momento y esfuerzo
        desviador intactos), con Q = σ|F|²/(ψ² τ). En BGK eso es exactamente

            C_i = 3 Q w_i (3|e_i|² − 2),

        que suma 2Q δ_αβ al segundo momento: una presión isótropa extra
        ∝ |∇ψ|² que desplaza ε de 0 a κσ.

        κ lo MEDÍ en vez de transcribirlo: la constante publicada (32σ en
        la normalización de Li et al.) usa otros pesos, otro G y otra
        escala de ψ. Con interfaces planas a 0.8 y 0.9 T_c, un único ε
        predice las DOS densidades de coexistencia a 5 cifras y ε/σ sale
        igual a ambas temperaturas: κ ≈ 5.2 (tests/test_coexistencia.py).
        σ = 0.33 ⇒ ε ≈ 1.7, que es lo que hace coincidir la coexistencia
        con Maxwell para esta EOS en 0.7-0.9 T_c."""
        if self.p.sigma_li == 0.0:
            return 0.0
        Q = self.p.sigma_li * (Fx * Fx + Fy * Fy) / (np.maximum(psi * psi, 1e-30) * self.p.tau)
        return FORMA_LI * Q

    def paso(self, n: int = 1) -> None:
        if self.motor == "numba":
            from aguacero import nucleo_numba

            for _ in range(n):
                nucleo_numba.paso(self)
        else:
            for _ in range(n):
                self._paso()

    def _paso(self) -> None:
        f, tau = self.f, self.p.tau
        fluido = self.fluido
        rho = f.sum(axis=0)
        rho_seguro = np.where(fluido, rho, 1.0)
        Fx_coh, Fy_coh, psi = self._fuerza_cohesion(rho)
        Fx = Fx_coh
        Fy = Fy_coh + np.where(fluido, rho * self.p.gravedad, 0.0)
        ux = np.where(fluido, ((f * EX).sum(axis=0) + 0.5 * Fx) / rho_seguro, 0.0)
        uy = np.where(fluido, ((f * EY).sum(axis=0) + 0.5 * Fy) / rho_seguro, 0.0)

        eu = EX * ux + EY * uy
        fuente_guo = (1.0 - 0.5 / tau) * W3 * (
            3.0 * ((EX - ux) * Fx + (EY - uy) * Fy) + 9.0 * eu * (EX * Fx + EY * Fy)
        )
        f_col = f - (f - equilibrio(rho, ux, uy)) / tau + fuente_guo + self._correccion_li(Fx_coh, Fy_coh, psi)
        f = np.where(fluido, f_col, f)

        f = propagar(f)
        solido = self.solido
        f[:, solido] = f[OPUESTO][:, solido]

        self.f = f
        self.rho, self.ux, self.uy = rho, ux, uy
        if not self.p.isotermico:
            self.T = paso_temperatura(
                self.T, rho, ux, uy, fluido, self.fuente, self.T_fuente, self.p.eos, self.p.chi
            )
        self.pasos += 1

    # ---------------------------------------------------------- observables

    def masa_total(self) -> float:
        """Suma sobre TODOS los nodos, paredes incluidas: con rebote completo
        las poblaciones que chocan con una pared pasan un paso "dentro" de
        ella antes de volver. Esta es la cantidad que se conserva exacta."""
        return float(self.f.sum())

    def presion(self) -> np.ndarray:
        """Presión termodinámica local p_EOS(ρ, T). En el seno de cada fase
        es LA presión; dentro de la interfaz (2-3 pixeles) la presión
        mecánica incluye además términos de gradiente, que aquí no se
        suman."""
        return np.where(self.fluido, self.p.eos.presion(self.rho, self.T), np.nan)

    def fraccion_liquido(self) -> np.ndarray:
        return np.clip((self.rho - self.rho_vapor) / (self.rho_liquido - self.rho_vapor), 0.0, 1.0)

    def energias(self) -> dict[str, float]:
        m = self.fluido
        rho = self.rho
        cinetica = 0.5 * rho * (self.ux**2 + self.uy**2)
        potencial = rho * self.p.gravedad * self.altura
        interna = self.p.eos.densidad_energia_interna(rho, self.T)
        e = {
            "cinetica": float(cinetica[m].sum()),
            "potencial": float(potencial[m].sum()),
            "interna": float(interna[m].sum()),
        }
        e["total"] = e["cinetica"] + e["potencial"] + e["interna"]
        return e

    # ------------------------------------------------------- herramientas

    def _poner_equilibrio(self, mascara, rho, ux=0.0, uy=0.0):
        mascara = mascara & self.fluido
        if not mascara.any():
            return
        antes = float(self.f[:, mascara].sum())
        n = int(mascara.sum())
        rho_arr = np.full(n, rho) if np.isscalar(rho) else rho[mascara]
        ux_arr = np.full(n, ux)
        uy_arr = np.full(n, uy)
        self.f[:, mascara] = equilibrio(rho_arr[None], ux_arr[None], uy_arr[None])[:, 0]
        self.masa_agregada += float(self.f[:, mascara].sum()) - antes

    def agregar_liquido(self, mascara, ux=0.0, uy=0.0, T_reducida: float | None = None):
        """Pone líquido en `mascara` con un borde suavizado de ~2 pixeles.

        Un escalón ρ_vapor → ρ_líquido de un pixel no es un estado de
        equilibrio: la interfaz de este modelo mide 3-4 pixeles, y relajar
        el escalón produce una onda de presión con |u| ~ 0.4 (Mach > 0.5)
        que puede tumbar la simulación. Con el perfil suavizado el
        transitorio baja a |u| ~ 0.05."""
        phi = gaussian_filter(mascara.astype(float), sigma=1.2, mode="nearest")
        rho_nueva = self.rho_vapor + (self.rho_liquido - self.rho_vapor) * np.clip(phi, 0.0, 1.0)
        rho_actual = self.f.sum(axis=0)
        donde = (rho_nueva > rho_actual + 1e-9) & (phi > 0.02) & self.fluido
        self._poner_equilibrio(donde, rho_nueva, ux, uy)
        if T_reducida is not None:
            self.T[donde] = T_reducida * self.p.eos.T_critica

    def quitar_liquido(self, mascara):
        self._poner_equilibrio(mascara, self.rho_vapor)

    def agregar_pared(self, mascara):
        nuevas = mascara & ~self.solido
        self.solido |= nuevas

    def quitar_pared(self, mascara, marco_protegido: bool = True):
        if marco_protegido:
            mascara = mascara.copy()
            mascara[0, :] = mascara[-1, :] = mascara[:, 0] = mascara[:, -1] = False
        liberadas = mascara & self.solido
        self.solido &= ~liberadas
        self.fuente &= ~liberadas
        self.T[liberadas] = self.T0
        self._poner_equilibrio(liberadas, self.rho_vapor)

    def agregar_fuente(self, mascara, T_reducida: float):
        """Placa a temperatura fija (calefactor si T > T0, enfriador si T < T0).
        Es sólida para el fluido y Dirichlet para la temperatura."""
        self.agregar_pared(mascara)
        self.fuente |= mascara
        self.T_fuente[mascara] = T_reducida * self.p.eos.T_critica
        self.T[mascara] = T_reducida * self.p.eos.T_critica
