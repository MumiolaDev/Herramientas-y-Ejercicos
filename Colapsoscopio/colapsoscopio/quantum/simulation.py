"""Orquestador: junta malla + potencial + condición inicial + solver, corre
la evolución temporal y devuelve una Trajectory lista para visualizar o
para verificar en un test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from colapsoscopio.quantum.core.grid import Grid1D
from colapsoscopio.quantum.core.hamiltonian import Hamiltonian1D
from colapsoscopio.quantum.core.state import WaveFunction
from colapsoscopio.quantum.initial_conditions.base import InitialCondition
from colapsoscopio.quantum.potentials.base import Potential
from colapsoscopio.quantum.solvers.base import Solver
from colapsoscopio.quantum.solvers.split_step import SplitStepSolver


@dataclass
class SimulationConfig:
    grid: Grid1D
    potential: Potential
    initial_condition: InitialCondition
    dt: float
    n_steps: int
    hbar: float = 1.0
    mass: float = 1.0
    guardar_cada: int = 1
    """Subsampleo: 1 = guarda todos los pasos, N = guarda 1 de cada N (útil
    cuando dt debe ser muy chico por estabilidad pero graficar cada paso
    sería redundante)."""
    solver_factory: type[Solver] = field(default=SplitStepSolver)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt debe ser positivo")
        if self.n_steps < 1:
            raise ValueError("n_steps debe ser >= 1")
        if self.guardar_cada < 1:
            raise ValueError("guardar_cada debe ser >= 1")


@dataclass
class Trajectory:
    """Resultado de una simulación: Psi(x,t) en los tiempos guardados, más
    los observables ya reducidos a series de tiempo (evita recalcularlos
    para cada frame de una animación).
    """

    grid: Grid1D
    tiempos: np.ndarray  # shape (n_snapshots,)
    estados: np.ndarray  # complejo, shape (n_snapshots, n_points)
    norma: np.ndarray  # shape (n_snapshots,)
    energia: np.ndarray  # shape (n_snapshots,)
    x_esperado: np.ndarray  # shape (n_snapshots,)
    hbar: float = 1.0
    mass: float = 1.0

    @property
    def n_snapshots(self) -> int:
        return len(self.tiempos)

    def densidad(self, i: int) -> np.ndarray:
        """|Psi(x, t_i)|^2."""
        return np.abs(self.estados[i]) ** 2

    def estado_en(self, i: int) -> WaveFunction:
        return WaveFunction(self.grid, self.estados[i])


class Simulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.hamiltonian = Hamiltonian1D(
            grid=config.grid, potential=config.potential, hbar=config.hbar, mass=config.mass
        )
        self.solver = config.solver_factory(self.hamiltonian)

    def run(self) -> Trajectory:
        cfg = self.config
        psi = cfg.initial_condition.construir(cfg.grid, cfg.potential, cfg.hbar, cfg.mass)

        n_snapshots = cfg.n_steps // cfg.guardar_cada + 1
        estados = np.empty((n_snapshots, cfg.grid.n_points), dtype=complex)
        tiempos = np.empty(n_snapshots)
        norma = np.empty(n_snapshots)
        energia = np.empty(n_snapshots)
        x_esperado = np.empty(n_snapshots)

        def registrar(idx: int, t: float, psi: WaveFunction) -> None:
            estados[idx] = psi.psi
            tiempos[idx] = t
            norma[idx] = psi.norma()
            energia[idx] = self.hamiltonian.valor_esperado_energia(psi)
            x_esperado[idx] = psi.valor_esperado_x()

        registrar(0, 0.0, psi)
        idx = 1
        t = 0.0
        for paso in range(1, cfg.n_steps + 1):
            psi = self.solver.paso(psi, cfg.dt)
            t += cfg.dt
            if paso % cfg.guardar_cada == 0:
                registrar(idx, t, psi)
                idx += 1

        return Trajectory(
            grid=cfg.grid,
            tiempos=tiempos,
            estados=estados,
            norma=norma,
            energia=energia,
            x_esperado=x_esperado,
            hbar=cfg.hbar,
            mass=cfg.mass,
        )
