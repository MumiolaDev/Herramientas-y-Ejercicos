"""Orquestador: junta sistema + estado inicial + integrador, corre la
evolución temporal y devuelve una Trajectory lista para visualizar o
verificar en un test. Misma idea que Simulation/Trajectory en Colapsoscopio,
en el dominio clásico.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caoscopio.core.integrator import RK4Integrator
from caoscopio.systems.base import DynamicalSystem


@dataclass
class SimulationConfig:
    sistema: DynamicalSystem
    estado_inicial: np.ndarray
    dt: float
    n_steps: int
    guardar_cada: int = 1
    integrador_factory: type = field(default=RK4Integrator)

    def __post_init__(self) -> None:
        self.estado_inicial = np.asarray(self.estado_inicial, dtype=float)
        if self.estado_inicial.shape != (self.sistema.dim,):
            raise ValueError(f"estado_inicial debe tener forma ({self.sistema.dim},)")
        if self.dt <= 0:
            raise ValueError("dt debe ser positivo")
        if self.n_steps < 1:
            raise ValueError("n_steps debe ser >= 1")
        if self.guardar_cada < 1:
            raise ValueError("guardar_cada debe ser >= 1")


@dataclass
class Trajectory:
    tiempos: np.ndarray  # shape (n_snapshots,)
    estados: np.ndarray  # shape (n_snapshots, dim)
    energia: np.ndarray  # shape (n_snapshots,) — NaN si el sistema no define energia()

    @property
    def n_snapshots(self) -> int:
        return len(self.tiempos)


class Simulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.integrador = config.integrador_factory(config.sistema)

    def run(self) -> Trajectory:
        cfg = self.config
        sistema = cfg.sistema
        tiene_energia = True

        n_snapshots = cfg.n_steps // cfg.guardar_cada + 1
        tiempos = np.empty(n_snapshots)
        estados = np.empty((n_snapshots, sistema.dim))
        energia = np.empty(n_snapshots)

        def registrar(idx: int, t: float, y: np.ndarray) -> None:
            nonlocal tiene_energia
            tiempos[idx] = t
            estados[idx] = y
            if tiene_energia:
                try:
                    energia[idx] = sistema.energia(y)
                except NotImplementedError:
                    tiene_energia = False
                    energia[idx] = np.nan
            else:
                energia[idx] = np.nan

        y = cfg.estado_inicial.copy()
        t = 0.0
        registrar(0, t, y)
        idx = 1
        for paso in range(1, cfg.n_steps + 1):
            y = self.integrador.paso(t, y, cfg.dt)
            t += cfg.dt
            if paso % cfg.guardar_cada == 0:
                registrar(idx, t, y)
                idx += 1

        return Trajectory(tiempos=tiempos, estados=estados, energia=energia)
