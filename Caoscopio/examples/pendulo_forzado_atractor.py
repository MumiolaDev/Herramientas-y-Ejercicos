"""Ejemplo: péndulo forzado y amortiguado, en régimen periódico (atractor
simple) y caótico (atractor extraño). Para cada régimen: el campo
vectorial congelado a la fase de forzado t≡0 (mod T_d), varias
trayectorias desde condiciones iniciales bien distintas convergiendo hacia
el mismo conjunto, y los puntos de la sección de Poincaré estroboscópica
(una muestra por período de forzado) que revelan la estructura del
atractor — un punto en el régimen periódico, una nube con estructura fina
en el caótico.

Corre:
    python examples/pendulo_forzado_atractor.py

Genera:
    salidas/pendulo_forzado_atractor.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from caoscopio import DrivenDampedPendulum, Simulation, SimulationConfig

CONDICIONES_INICIALES = [
    [0.2, 0.0], [2.5, 1.0], [-1.5, -2.0], [3.0, 0.5], [-2.8, 1.5], [1.0, -2.5],
]


def simular_fino(sistema: DrivenDampedPendulum, estado_inicial, n_periodos: int, pasos_por_periodo: int = 300, guardar_cada: int = 6):
    """Trayectoria a resolución fina (varios puntos por período, no solo
    uno) — para poder dibujar la curva continua de caída hacia el
    atractor, no solo su muestreo estroboscópico."""
    dt = sistema.periodo_forzado / pasos_por_periodo
    config = SimulationConfig(
        sistema=sistema, estado_inicial=np.array(estado_inicial), dt=dt,
        n_steps=pasos_por_periodo * n_periodos, guardar_cada=guardar_cada,
    )
    return Simulation(config).run()


def graficar_panel(ax, sistema: DrivenDampedPendulum, titulo: str, n_periodos: int, n_transiente: int):
    theta_malla, omega_malla = np.meshgrid(np.linspace(-4, 4, 22), np.linspace(-4, 4, 22))
    dtheta, domega = sistema.campo_vectorial(theta_malla, omega_malla, t=0.0)
    norma = np.hypot(dtheta, domega) + 1e-9
    ax.quiver(theta_malla, omega_malla, dtheta / norma, domega / norma, color="#c7ccc8", scale=28, width=0.003)

    pasos_por_periodo = 300
    for ic in CONDICIONES_INICIALES:
        traj = simular_fino(sistema, ic, n_periodos, pasos_por_periodo)
        theta = (traj.estados[:, 0] + np.pi) % (2 * np.pi) - np.pi
        omega = traj.estados[:, 1]

        # curva continua de caída hacia el atractor, solo el transiente
        idx_transiente = int(n_transiente * pasos_por_periodo / 6)  # guardar_cada=6
        ax.plot(theta[:idx_transiente], omega[:idx_transiente], "-", lw=0.5, alpha=0.35, color="#2a78d6")

        # sección de Poincaré (una muestra por período) ya sin transiente
        idx_poincare = np.arange(0, len(theta), pasos_por_periodo // 6)
        idx_poincare = idx_poincare[idx_poincare >= idx_transiente]
        ax.plot(theta[idx_poincare], omega[idx_poincare], ".", ms=2.6, alpha=0.7, color="#eb6834")

    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_xlabel("θ (rad, envuelto a [-π,π])"); ax.set_ylabel("ω (rad/s)")
    ax.set_title(titulo, fontsize=10)


def main() -> None:
    periodico = DrivenDampedPendulum(A=0.9)
    caotico = DrivenDampedPendulum(A=1.15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))
    graficar_panel(ax1, periodico, "A=0.9 — atractor periódico (ciclo límite)", n_periodos=250, n_transiente=100)
    graficar_panel(ax2, caotico, "A=1.15 — atractor extraño (caótico)", n_periodos=400, n_transiente=100)
    fig.suptitle("Sección de Poincaré estroboscópica sobre el campo vectorial congelado (t≡0 mod T_d)", fontsize=10)
    fig.tight_layout()

    os.makedirs("salidas", exist_ok=True)
    fig.savefig("salidas/pendulo_forzado_atractor.png", dpi=140)
    print("guardado: salidas/pendulo_forzado_atractor.png")


if __name__ == "__main__":
    main()
