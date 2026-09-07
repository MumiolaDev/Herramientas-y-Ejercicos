"""Ejemplo: paquete de onda gaussiano desplazado del centro de un oscilador
armónico. Al no ser un autoestado, oscila de ida y vuelta reproduciendo (de
forma aproximada) el movimiento clásico x(t) = x0 cos(omega t), sin
deformarse apenas —el caso límite exacto es un "estado coherente"—.

Corre:
    python examples/oscilador_armonico.py

Genera:
    salidas/oscilador_armonico.gif
    salidas/oscilador_armonico_ascii.txt
"""

from __future__ import annotations

import os

from colapsoscopio import Grid1D, Simulation, SimulationConfig
from colapsoscopio.quantum.initial_conditions import GaussianPacket
from colapsoscopio.quantum.potentials import HarmonicOscillator
from colapsoscopio.quantum.visualization import AsciiAnimator, MatplotlibAnimator


def main() -> None:
    omega = 1.0
    mass = 1.0
    hbar = 1.0
    longitud_caracteristica = (hbar / (mass * omega)) ** 0.5

    # caja bien ancha frente a la longitud característica del oscilador,
    # para que la condición periódica en los bordes no contamine la dinámica
    x_max = 10 * longitud_caracteristica
    grid = Grid1D(x_min=-x_max, x_max=x_max, n_points=1024, boundary="periodic")
    potential = HarmonicOscillator(omega=omega, mass=mass)

    # paquete de mínima incertidumbre desplazado x0, en reposo (k0=0): es
    # (aproximadamente) el estado coherente |alpha> con alpha real
    ic = GaussianPacket(x0=3 * longitud_caracteristica, sigma=longitud_caracteristica, k0=0.0)

    periodo_clasico = 2 * 3.141592653589793 / omega
    config = SimulationConfig(
        grid=grid,
        potential=potential,
        initial_condition=ic,
        dt=periodo_clasico / 4000,
        n_steps=8000,  # ~2 períodos clásicos
        guardar_cada=20,
        hbar=hbar,
        mass=mass,
    )
    traj = Simulation(config).run()

    print(f"norma inicial={traj.norma[0]:.8f}  norma final={traj.norma[-1]:.8f}")
    print(f"<E> inicial={traj.energia[0]:.5f}  <E> final={traj.energia[-1]:.5f}")
    print(f"periodo clásico esperado: {periodo_clasico:.4f}")

    os.makedirs("salidas", exist_ok=True)

    animador = MatplotlibAnimator(traj, potencial_x=potential(grid.x))
    animador.guardar("salidas/oscilador_armonico.gif", fps=25)
    print("guardado: salidas/oscilador_armonico.gif")

    ascii_animador = AsciiAnimator(traj)
    ascii_animador.guardar_texto("salidas/oscilador_armonico_ascii.txt")
    print("guardado: salidas/oscilador_armonico_ascii.txt")


if __name__ == "__main__":
    main()
