"""Ejemplo: el atractor de Lorenz — la "mariposa" más famosa de la teoría
del caos. Dos trayectorias con condiciones iniciales que difieren en
10⁻⁵ en x(0): idénticas al ojo al principio, irreconciliables después de
recorrer el atractor un rato — el mismo efecto mariposa de
examples/pendulo_doble_sensibilidad.py, pero acá el propio nombre del
fenómeno viene de este sistema (Lorenz usó la metáfora en una charla de
1972 titulada "¿el aleteo de una mariposa en Brasil desencadena un tornado
en Texas?", refiriéndose exactamente a esta ecuación).

Corre:
    python examples/lorenz_mariposa.py

Genera:
    salidas/lorenz_mariposa.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from caoscopio import Lorenz, Simulation, SimulationConfig


def main() -> None:
    sistema = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)

    estado_a = np.array([1.0, 1.0, 1.0])
    estado_b = estado_a + np.array([1e-5, 0.0, 0.0])

    config_a = SimulationConfig(sistema, estado_a, dt=1e-3, n_steps=20000, guardar_cada=10)
    config_b = SimulationConfig(sistema, estado_b, dt=1e-3, n_steps=20000, guardar_cada=10)
    traj_a = Simulation(config_a).run()
    traj_b = Simulation(config_b).run()

    separacion = np.linalg.norm(traj_a.estados - traj_b.estados, axis=1)
    print(f"|Δ| inicial = {separacion[0]:.1e}   |Δ| final = {separacion[-1]:.3f}")
    print(f"divergencia teórica del campo (contracción de volumen): {sistema.divergencia:.4f}")

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(projection="3d")
    ax.plot(*traj_a.estados.T, lw=0.5, color="#2a78d6", alpha=0.85, label="trayectoria A")
    ax.plot(*traj_b.estados.T, lw=0.5, color="#eb6834", alpha=0.55, label=f"trayectoria B (Δx₀={1e-5:.0e})")
    for pf in sistema.puntos_fijos():
        ax.scatter(*pf, color="#1baf7a", s=25, depthshade=False)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Atractor de Lorenz (σ=10, ρ=28, β=8/3) — puntos verdes: equilibrios exactos")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    os.makedirs("salidas", exist_ok=True)
    fig.savefig("salidas/lorenz_mariposa.png", dpi=140)
    print("guardado: salidas/lorenz_mariposa.png")

    # la vista 3D por sí sola no deja ver bien la divergencia (ambas
    # trayectorias viven en el mismo atractor acotado, así que se cruzan
    # y se acercan por casualidad una y otra vez pese a haber divergido
    # hace rato) — el crecimiento exponencial inicial se ve mejor en
    # escala log, igual que en examples/pendulo_doble_sensibilidad.py
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    ax2.semilogy(traj_a.tiempos, separacion, color="#2a78d6", lw=1.2)
    ax2.set_xlabel("t"); ax2.set_ylabel("|Δ| (escala log)")
    ax2.set_title("Separación entre las dos trayectorias — crecimiento exponencial, luego satura")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("salidas/lorenz_divergencia.png", dpi=130)
    print("guardado: salidas/lorenz_divergencia.png")


if __name__ == "__main__":
    main()
