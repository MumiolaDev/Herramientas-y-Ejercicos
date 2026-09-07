"""Ejemplo de validación: un autoestado del Hamiltoniano es estacionario, así
que |Psi(x,t)|^2 no debería cambiar en absoluto en el tiempo (solo Psi
acumula una fase global e^{-i E_n t/hbar}). Esto se corre para el pozo
infinito y el oscilador armónico, y compara la fase numérica contra la
predicción analítica.

Corre:
    python examples/validar_autoestados.py
"""

from __future__ import annotations

import numpy as np

from colapsoscopio import Grid1D, Simulation, SimulationConfig
from colapsoscopio.quantum.initial_conditions import Eigenstate
from colapsoscopio.quantum.potentials import HarmonicOscillator, InfiniteWell


def validar_pozo_infinito(n: int = 2) -> None:
    grid = Grid1D(x_min=0.0, x_max=10.0, n_points=400, boundary="dirichlet")
    potential = InfiniteWell()
    ic = Eigenstate(n=n)

    config = SimulationConfig(
        grid=grid, potential=potential, initial_condition=ic, dt=1e-3, n_steps=500
    )
    traj = Simulation(config).run()

    densidad_inicial = traj.densidad(0)
    error_densidad = np.max(np.abs(traj.densidad(-1) - densidad_inicial))
    print(f"[pozo infinito, n={n}]  E_n analítico = {ic.energia:.6f}")
    print(f"  <E> numérico (t=0)   = {traj.energia[0]:.6f}")
    print(f"  <E> numérico (t=fin) = {traj.energia[-1]:.6f}")
    print(f"  max |Δ densidad| en todo el intervalo = {error_densidad:.3e}  (debería ser ~0)")


def validar_oscilador_armonico(n: int = 3) -> None:
    omega, mass, hbar = 1.0, 1.0, 1.0
    l0 = (hbar / (mass * omega)) ** 0.5
    grid = Grid1D(x_min=-10 * l0, x_max=10 * l0, n_points=800, boundary="periodic")
    potential = HarmonicOscillator(omega=omega, mass=mass)
    ic = Eigenstate(n=n)

    config = SimulationConfig(
        grid=grid,
        potential=potential,
        initial_condition=ic,
        dt=1e-3,
        n_steps=500,
        hbar=hbar,
        mass=mass,
    )
    traj = Simulation(config).run()

    densidad_inicial = traj.densidad(0)
    error_densidad = np.max(np.abs(traj.densidad(-1) - densidad_inicial))
    print(f"[oscilador armónico, n={n}]  E_n analítico = {ic.energia:.6f}")
    print(f"  <E> numérico (t=0)   = {traj.energia[0]:.6f}")
    print(f"  <E> numérico (t=fin) = {traj.energia[-1]:.6f}")
    print(f"  max |Δ densidad| en todo el intervalo = {error_densidad:.3e}  (debería ser ~0)")


if __name__ == "__main__":
    validar_pozo_infinito()
    print()
    validar_oscilador_armonico()
