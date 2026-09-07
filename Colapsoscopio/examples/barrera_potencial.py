"""Ejemplo: efecto túnel. Un paquete gaussiano con energía media *menor* que
la altura de una barrera rectangular llega, en parte se refleja (como
predice la física clásica) y en parte aparece del otro lado (lo que la
física clásica prohíbe de plano).

Corre:
    python examples/barrera_potencial.py

Genera:
    salidas/barrera_potencial.gif
    salidas/barrera_potencial_ascii.txt
"""

from __future__ import annotations

import os

from colapsoscopio import Grid1D, Simulation, SimulationConfig
from colapsoscopio.quantum.initial_conditions import GaussianPacket
from colapsoscopio.quantum.potentials import PotentialBarrier
from colapsoscopio.quantum.visualization import AsciiAnimator, MatplotlibAnimator


def main() -> None:
    hbar = mass = 1.0
    k0, sigma0, x0 = 1.0, 2.0, -8.0
    v0, centro, ancho = 1.0, 0.0, 1.5

    barrera = PotentialBarrier(v0=v0, centro=centro, ancho=ancho)
    grid = Grid1D(x_min=-25.0, x_max=25.0, n_points=1024, boundary="periodic")
    ic = GaussianPacket(x0=x0, sigma=sigma0, k0=k0)

    config = SimulationConfig(
        grid=grid, potential=barrera, initial_condition=ic, dt=5e-4, n_steps=36000, guardar_cada=180
    )
    traj = Simulation(config).run()

    # energía media del paquete (no es monocromático: <E> = <p^2>/2m con
    # <p^2> = hbar^2 k0^2 + hbar^2/(4 sigma0^2) para un gaussiano de mínima
    # incertidumbre), y el T(E) *teórico* de la barrera estacionaria en esa
    # energía — solo referencia, porque el paquete trae una distribución de
    # energías, no una sola.
    energia_media = 0.5 * (k0**2 + 1 / (4 * sigma0**2))
    t_teorico = barrera.transmision_teorica(energia_media, hbar=hbar, mass=mass)

    x = grid.x
    densidad_final = traj.densidad(-1)
    lado_derecho = x > (centro + ancho / 2)
    lado_izquierdo = x < (centro - ancho / 2)
    p_transmitida = grid.integrate(densidad_final * lado_derecho)
    p_reflejada = grid.integrate(densidad_final * lado_izquierdo)

    print(f"norma inicial={traj.norma[0]:.8f}  norma final={traj.norma[-1]:.8f}")
    print(f"<E> inicial={traj.energia[0]:.5f}  <E> final={traj.energia[-1]:.5f}  (V0={v0})")
    print(f"T teórico (barrera estacionaria, en <E>={energia_media:.3f}) = {t_teorico:.4f}")
    print(f"P transmitida (numérica, paquete real)                       = {p_transmitida:.4f}")
    print(f"P reflejada  (numérica)                                      = {p_reflejada:.4f}")
    print("(el desacuerdo entre las dos filas de T es esperado: el paquete no es")
    print(" monocromático, así que su transmisión es un promedio de T(E) sobre su")
    print(" propio espectro de energías, no T(<E>) evaluado en la energía media)")

    os.makedirs("salidas", exist_ok=True)

    animador = MatplotlibAnimator(traj, potencial_x=barrera(grid.x))
    animador.guardar("salidas/barrera_potencial.gif", fps=25)
    print("guardado: salidas/barrera_potencial.gif")

    ascii_animador = AsciiAnimator(traj)
    ascii_animador.guardar_texto("salidas/barrera_potencial_ascii.txt")
    print("guardado: salidas/barrera_potencial_ascii.txt")


if __name__ == "__main__":
    main()
