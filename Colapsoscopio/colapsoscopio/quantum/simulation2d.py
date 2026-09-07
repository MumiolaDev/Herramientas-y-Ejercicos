"""Orquestador 2D — análogo a simulation.py (1D), ver ese docstring."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from colapsoscopio.quantum.core.grid2d import Grid2D
from colapsoscopio.quantum.core.hamiltonian2d import Hamiltonian2D
from colapsoscopio.quantum.core.state2d import WaveFunction2D
from colapsoscopio.quantum.initial_conditions.gaussian_packet_2d import GaussianPacket2D
from colapsoscopio.quantum.potentials.base2d import Potential2D
from colapsoscopio.quantum.solvers.split_step_2d import SplitStepSolver2D


@dataclass
class SimulationConfig2D:
    grid: Grid2D
    potential: Potential2D
    initial_condition: GaussianPacket2D
    dt: float
    n_steps: int
    hbar: float = 1.0
    mass: float = 1.0
    guardar_cada: int = 1
    solver_factory: type = field(default=SplitStepSolver2D)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt debe ser positivo")
        if self.n_steps < 1:
            raise ValueError("n_steps debe ser >= 1")
        if self.guardar_cada < 1:
            raise ValueError("guardar_cada debe ser >= 1")


@dataclass
class Trajectory2D:
    grid: Grid2D
    tiempos: np.ndarray  # shape (n_snapshots,)
    estados: np.ndarray  # complejo, shape (n_snapshots, n_x, n_y)
    norma: np.ndarray
    energia: np.ndarray
    x_esperado: np.ndarray
    y_esperado: np.ndarray
    hbar: float = 1.0
    mass: float = 1.0

    @property
    def n_snapshots(self) -> int:
        return len(self.tiempos)

    def densidad(self, i: int) -> np.ndarray:
        return np.abs(self.estados[i]) ** 2

    def estado_en(self, i: int) -> WaveFunction2D:
        return WaveFunction2D(self.grid, self.estados[i])


class Simulation2D:
    def __init__(self, config: SimulationConfig2D):
        self.config = config
        self.hamiltonian = Hamiltonian2D(
            grid=config.grid, potential=config.potential, hbar=config.hbar, mass=config.mass
        )
        self.solver = config.solver_factory(self.hamiltonian)

    def run(self) -> Trajectory2D:
        cfg = self.config
        psi = cfg.initial_condition.construir(cfg.grid, cfg.potential, cfg.hbar, cfg.mass)

        n_snapshots = cfg.n_steps // cfg.guardar_cada + 1
        estados = np.empty((n_snapshots, cfg.grid.n_x, cfg.grid.n_y), dtype=complex)
        tiempos = np.empty(n_snapshots)
        norma = np.empty(n_snapshots)
        energia = np.empty(n_snapshots)
        x_esperado = np.empty(n_snapshots)
        y_esperado = np.empty(n_snapshots)

        def registrar(idx: int, t: float, psi: WaveFunction2D) -> None:
            estados[idx] = psi.psi
            tiempos[idx] = t
            norma[idx] = psi.norma()
            energia[idx] = self.hamiltonian.valor_esperado_energia(psi)
            x_esperado[idx] = psi.valor_esperado_x()
            y_esperado[idx] = psi.valor_esperado_y()

        registrar(0, 0.0, psi)
        idx = 1
        t = 0.0
        for paso in range(1, cfg.n_steps + 1):
            psi = self.solver.paso(psi, cfg.dt)
            t += cfg.dt
            if paso % cfg.guardar_cada == 0:
                registrar(idx, t, psi)
                idx += 1

        return Trajectory2D(
            grid=cfg.grid,
            tiempos=tiempos,
            estados=estados,
            norma=norma,
            energia=energia,
            x_esperado=x_esperado,
            y_esperado=y_esperado,
            hbar=cfg.hbar,
            mass=cfg.mass,
        )
