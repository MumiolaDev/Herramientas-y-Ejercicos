"""Método split-operator (split-step espectral), de 2do orden en dt.

La idea (Feit, Fleck & Steiger 1982; De Raedt 1987) es que para dt pequeño

    e^{-i H dt/hbar} = e^{-i (T+V) dt/hbar}
                      ≈ e^{-i V dt/(2 hbar)} e^{-i T dt/hbar} e^{-i V dt/(2 hbar)}
                        + O(dt^3)

es decir, aplicar V/2 en el espacio donde V es diagonal (posición), pasar a
la base donde T es diagonal (Fourier si la malla es periódica, seno-discreta
si es Dirichlet — ver Grid1D.transformar_ida/vuelta), aplicar T completo ahí,
volver y aplicar V/2 otra vez. Cada uno de esos tres factores es una
exponencial de un operador diagonal, o sea un producto elemento a elemento
por una fase compleja: no se ensambla ni se invierte ninguna matriz, y cada
paso es exactamente unitario salvo el error de redondeo de punto flotante
(la norma se conserva a machine precision, no solo aproximadamente). Esa
propiedad —unitariedad exacta independiente de dt— es la razón por la que
este método es el estándar de facto para integrar la TDSE, y la que permite
extenderlo sin cambios conceptuales a 2D/3D o a coordenadas radiales para el
átomo de hidrógeno: basta con tener una transformada donde T sea diagonal.
"""

from __future__ import annotations

import numpy as np

from colapsoscopio.quantum.core.hamiltonian import Hamiltonian1D
from colapsoscopio.quantum.core.state import WaveFunction
from colapsoscopio.quantum.solvers.base import Solver


class SplitStepSolver(Solver):
    def __init__(self, hamiltonian: Hamiltonian1D):
        self.hamiltonian = hamiltonian
        self._v_medio_cache: dict[float, np.ndarray] = {}
        self._t_completo_cache: dict[float, np.ndarray] = {}

    def _fase_v_medio(self, dt: float) -> np.ndarray:
        if dt not in self._v_medio_cache:
            v = self.hamiltonian.potencial_x()
            self._v_medio_cache[dt] = np.exp(-1j * v * dt / (2 * self.hamiltonian.hbar))
        return self._v_medio_cache[dt]

    def _fase_t_completo(self, dt: float) -> np.ndarray:
        if dt not in self._t_completo_cache:
            t = self.hamiltonian.energia_cinetica_k()
            self._t_completo_cache[dt] = np.exp(-1j * t * dt / self.hamiltonian.hbar)
        return self._t_completo_cache[dt]

    def paso(self, psi: WaveFunction, dt: float) -> WaveFunction:
        grid = self.hamiltonian.grid
        fase_v = self._fase_v_medio(dt)
        fase_t = self._fase_t_completo(dt)

        psi_x = fase_v * psi.psi
        psi_k = grid.transformar_ida(psi_x)
        psi_k = fase_t * psi_k
        psi_x = grid.transformar_vuelta(psi_k)
        psi_x = fase_v * psi_x

        return WaveFunction(grid, psi_x)
