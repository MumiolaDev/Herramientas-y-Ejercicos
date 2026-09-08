"""Ejemplo: sensibilidad a condiciones iniciales — el "efecto mariposa" en
carne propia. Dos péndulos dobles idénticos, en el mismo régimen caótico
de examples/pendulo_doble_caotico.py, con una diferencia inicial en θ1 de
apenas 10⁻³ rad (~0.057°, invisible a simple vista en el cuadro inicial).
Ambos se integran con el mismo método y el mismo dt —la única diferencia
es esa condición inicial— y aun así terminan en configuraciones
completamente distintas.

Esto no calcula el exponente de Lyapunov (queda en el roadmap: eso pide
renormalizar la separación periódicamente y promediar su tasa de
crecimiento) — solo muestra la separación angular |Δθ1(t)| creciendo, que
es la observación cruda que motiva calcularlo.

Corre:
    python examples/pendulo_doble_sensibilidad.py

Genera:
    salidas/pendulo_doble_sensibilidad.gif
    salidas/pendulo_doble_sensibilidad_divergencia.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from caoscopio import DoublePendulum, Simulation, SimulationConfig
from caoscopio.visualization import PendulumAnimator


def main() -> None:
    sistema = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
    estado_a = np.array([2.4, 0.0, 2.4, 0.0])
    estado_b = estado_a.copy()
    estado_b[0] += 1e-3  # perturbación mínima en theta1

    dt, n_steps, guardar_cada = 1e-3, 20000, 60
    traj_a = Simulation(SimulationConfig(sistema, estado_a, dt, n_steps, guardar_cada)).run()
    traj_b = Simulation(SimulationConfig(sistema, estado_b, dt, n_steps, guardar_cada)).run()

    delta_theta1 = np.abs(traj_a.estados[:, 0] - traj_b.estados[:, 0])
    print(f"|Δθ1| inicial = {delta_theta1[0]:.2e}   |Δθ1| final = {delta_theta1[-1]:.4f} rad")
    print(f"(creció un factor {delta_theta1[-1]/delta_theta1[0]:.1e} en t={traj_a.tiempos[-1]:.1f}s)")

    os.makedirs("salidas", exist_ok=True)

    animador = PendulumAnimator(sistema, traj_a, trayectoria_gemela=traj_b)
    animador.guardar("salidas/pendulo_doble_sensibilidad.gif", fps=30)
    print("guardado: salidas/pendulo_doble_sensibilidad.gif")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogy(traj_a.tiempos, delta_theta1, color="#2a78d6", lw=1.5)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|Δθ₁| (rad, escala log)")
    ax.set_title("Separación angular entre dos condiciones iniciales casi idénticas (Δθ₁(0)=10⁻³)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("salidas/pendulo_doble_sensibilidad_divergencia.png", dpi=130)
    print("guardado: salidas/pendulo_doble_sensibilidad_divergencia.png")


if __name__ == "__main__":
    main()
