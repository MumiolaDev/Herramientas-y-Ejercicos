"""Ejemplo 2D: el experimento de la doble rendija con un paquete de onda.
Un paquete gaussiano incide sobre una pantalla opaca con dos aberturas; del
otro lado emerge el patrón de interferencia característico —franjas de
máximos y mínimos, no dos manchas separadas como predeciría la intuición
clásica de "partícula que pasa por una rendija u otra"—.

El corte de observación importa: por debajo de la distancia de Rayleigh
L_R ~ d²/λ (d = separación entre rendijas, λ = 2π/k0 la longitud de onda de
De Broglie) se está en campo cercano (Fresnel) — el patrón todavía no se
separó en franjas limpias. Este ejemplo propaga varias veces L_R (campo
lejano / Fraunhofer) para que el patrón salga nítido: con d=4, k0=4,
L_R≈10.2, y se observa hasta x=25 (~2.5×L_R).

Esta es la versión "de alta fidelidad": malla fina, dominio grande, tiempo
de cómputo real de varios minutos (ver el comentario de tiempos más abajo).
Para iterar rápido durante desarrollo, bajar n_x/n_y y el dominio reduce el
costo cuadráticamente sin cambiar la física cualitativa del patrón (eso ya
se ve incluso a resolución baja; lo que exige más malla es reproducir el
campo lejano completo con buen detalle, no la existencia del patrón).

Corre:
    python examples/doble_rendija.py

Genera:
    salidas/doble_rendija.gif
    salidas/doble_rendija_ascii.txt
"""

from __future__ import annotations

import os
import time

from colapsoscopio import Grid2D, Simulation2D, SimulationConfig2D
from colapsoscopio.quantum.initial_conditions import GaussianPacket2D
from colapsoscopio.quantum.potentials import DoubleSlit
from colapsoscopio.quantum.visualization import AsciiAnimator2D, MatplotlibAnimator2D


def main() -> None:
    # malla de ~656 000 puntos: en este proyecto, a esa resolución, cada
    # paso de split-operator (dos FFT2) toma ~25 ms — medido, no estimado
    # (ver el benchmark en la conversación que acompaña este commit). Con
    # 8250 pasos, el cómputo completo toma minutos, no segundos.
    grid = Grid2D(x_min=-35.0, x_max=35.0, n_x=875, y_min=-30.0, y_max=30.0, n_y=750)
    potential = DoubleSlit(v0=40.0, x_pantalla=0.0, grosor=0.5, separacion=4.0, ancho_rendija=0.8)

    # sigma_y grande (frente ancho en y): así el paquete ilumina ambas
    # rendijas de forma parecida a una onda plana, en vez de pasar
    # esencialmente por una sola.
    ic = GaussianPacket2D(x0=-8.0, y0=0.0, sigma_x=1.5, sigma_y=3.0, kx0=4.0, ky0=0.0)

    config = SimulationConfig2D(
        grid=grid, potential=potential, initial_condition=ic, dt=1e-3, n_steps=8250, guardar_cada=50
    )

    print(f"malla: {grid.n_x}x{grid.n_y} = {grid.n_x * grid.n_y} puntos, dx={grid.dx:.4f}")
    t0 = time.time()
    traj = Simulation2D(config).run()
    print(f"tiempo de cómputo: {time.time() - t0:.1f}s")

    print(f"norma inicial={traj.norma[0]:.8f}  norma final={traj.norma[-1]:.8f}")
    print(f"<E> inicial={traj.energia[0]:.5f}  <E> final={traj.energia[-1]:.5f}")

    os.makedirs("salidas", exist_ok=True)

    animador = MatplotlibAnimator2D(traj, potencial_xy=potential(*grid.meshgrid()))
    animador.guardar("salidas/doble_rendija.gif", fps=20)
    print("guardado: salidas/doble_rendija.gif")

    ascii_animador = AsciiAnimator2D(traj)
    ascii_animador.guardar_texto("salidas/doble_rendija_ascii.txt")
    print("guardado: salidas/doble_rendija_ascii.txt")


if __name__ == "__main__":
    main()
