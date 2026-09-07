"""Split-operator 2D: idéntico en espíritu a solvers/split_step.py (1D) — ver
ese docstring para la derivación — solo que V/2, T y V/2 son ahora arrays 2D
y la transformada es FFT2 en vez de FFT/DST 1D. Ninguna idea nueva: es la
misma prueba de que el método no depende de la dimensión, solo de tener una
transformada donde T sea diagonal.
"""

from __future__ import annotations

import numpy as np

from colapsoscopio.quantum.core.hamiltonian2d import Hamiltonian2D
from colapsoscopio.quantum.core.state2d import WaveFunction2D


class SplitStepSolver2D:
    def __init__(self, hamiltonian: Hamiltonian2D):
        self.hamiltonian = hamiltonian
        self._v_medio_cache: dict[float, np.ndarray] = {}
        self._t_completo_cache: dict[float, np.ndarray] = {}

    def _fase_v_medio(self, dt: float) -> np.ndarray:
        if dt not in self._v_medio_cache:
            v = self.hamiltonian.potencial_xy()
            self._v_medio_cache[dt] = np.exp(-1j * v * dt / (2 * self.hamiltonian.hbar))
        return self._v_medio_cache[dt]

    def _fase_t_completo(self, dt: float) -> np.ndarray:
        if dt not in self._t_completo_cache:
            t = self.hamiltonian.energia_cinetica_k()
            self._t_completo_cache[dt] = np.exp(-1j * t * dt / self.hamiltonian.hbar)
        return self._t_completo_cache[dt]

    def paso(self, psi: WaveFunction2D, dt: float) -> WaveFunction2D:
        grid = self.hamiltonian.grid
        fase_v = self._fase_v_medio(dt)
        fase_t = self._fase_t_completo(dt)

        psi_x = fase_v * psi.psi
        psi_k = grid.transformar_ida(psi_x)
        psi_k = fase_t * psi_k
        psi_x = grid.transformar_vuelta(psi_k)
        psi_x = fase_v * psi_x

        return WaveFunction2D(grid, psi_x)
