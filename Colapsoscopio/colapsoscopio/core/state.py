"""Estado cuántico Psi(x) sobre una malla, y sus observables derivados."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colapsoscopio.core.grid import Grid1D


@dataclass
class WaveFunction:
    """Función de onda discretizada psi_i = Psi(x_i) sobre una Grid1D.

    No se normaliza automáticamente al construirla: usa `normalizada()` para
    obtener una copia con norma 1, o `norma()` para chequear qué tan lejos
    está (útil como diagnóstico de la simulación: la norma debiera
    conservarse exactamente si el propagador es unitario).
    """

    grid: Grid1D
    psi: np.ndarray  # complejo, shape (grid.n_points,)

    def __post_init__(self) -> None:
        self.psi = np.asarray(self.psi, dtype=complex)
        if self.psi.shape != (self.grid.n_points,):
            raise ValueError("psi debe tener shape (grid.n_points,)")

    def densidad(self) -> np.ndarray:
        """|Psi(x)|^2, densidad de probabilidad."""
        return np.abs(self.psi) ** 2

    def norma(self) -> float:
        """Integral de |Psi|^2 dx. Debe ser ~1 para un estado físico."""
        return self.grid.integrate(self.densidad())

    def normalizada(self) -> "WaveFunction":
        n = self.norma()
        if n <= 0:
            raise ValueError("no se puede normalizar un estado de norma nula")
        return WaveFunction(self.grid, self.psi / np.sqrt(n))

    def valor_esperado_x(self) -> float:
        """<x> = integral x |Psi(x)|^2 dx."""
        return self.grid.integrate(self.grid.x * self.densidad())

    def valor_esperado_p(self, hbar: float = 1.0) -> float:
        """<p> = -i hbar <Psi| d/dx |Psi>, con derivada centrada de 2do orden.

        Se calcula en el espacio real (no en la base espectral de la malla)
        porque es la definición que sirve para ambas condiciones de frontera:
        en un pozo infinito la base de energía es senoidal y solo "ve" |k|,
        así que estimar <p> desde |Psi(k)|^2 perdería el signo (la dirección
        del movimiento). Con boundary="periodic" la derivada usa wraparound,
        consistente con la periodicidad de la malla; con "dirichlet" se
        extiende psi con ceros en los bordes fijos (donde Psi=0 por hipótesis).
        """
        if self.grid.boundary == "periodic":
            dpsi_dx = (np.roll(self.psi, -1) - np.roll(self.psi, 1)) / (2 * self.grid.dx)
        else:
            psi_ext = np.concatenate(([0j], self.psi, [0j]))
            dpsi_dx = (psi_ext[2:] - psi_ext[:-2]) / (2 * self.grid.dx)
        integrando = np.conj(self.psi) * dpsi_dx
        valor = -1j * hbar * np.sum(integrando) * self.grid.dx
        return float(np.real(valor))

    def valor_esperado(self, operador_diagonal: np.ndarray) -> float:
        """<A> para un operador diagonal en x (p.ej. el potencial V(x))."""
        return self.grid.integrate(operador_diagonal * self.densidad())
