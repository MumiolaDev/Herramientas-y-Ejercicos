"""Curva de coexistencia líquido-vapor: la simulación contra la teoría.

Para cada temperatura se deja relajar una losa de líquido (interfaz
plana, sin gravedad, isotérmica) y se miden las densidades de ambas
fases. Se comparan con

  - la construcción de Maxwell de la EOS de Carnahan-Starling (la
    termodinámica "de verdad"), y
  - la condición de equilibrio mecánico del pseudopotencial con ε = κσ
    (lo que el esquema discreto DEBE dar, ver aguacero/eos.py).

Corre:
    python examples/diagrama_de_fases.py

Genera:
    salidas/diagrama_de_fases.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aguacero.mundo import Mundo, Parametros


def losa(T_reducida: float, sigma_li: float):
    m = Mundo(128, 4, Parametros(T_reducida=T_reducida, gravedad=0.0, sigma_li=sigma_li, isotermico=True), marco=False)
    _, col = np.mgrid[0:4, 0:128]
    m.agregar_liquido((col >= 32) & (col < 96))
    m.paso(25000)
    return m, m.rho[0, 0], m.rho[0, 64]


def main() -> None:
    temps = np.array([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    eos = Parametros().eos
    T_fino = np.linspace(0.6, 0.97, 40)
    maxwell = np.array([eos.coexistencia(t * eos.T_critica, epsilon=None) for t in T_fino])
    ax.plot(maxwell[:, 0], T_fino, "k-", lw=1.2, label="Maxwell (Carnahan-Starling)")
    ax.plot(maxwell[:, 1], T_fino, "k-", lw=1.2)

    for sigma, color, etiqueta in [(0.0, "#eb6834", "simulación, Guo puro (σ=0)"), (Parametros().sigma_li, "#2a78d6", "simulación, con corrección de Li (σ=0.33)")]:
        medidos = []
        temps_validas = temps if sigma > 0 else temps[temps >= 0.8]  # con σ=0 no hay coexistencia bajo ~0.75 T_c
        for t in temps_validas:
            _, rg, rl = losa(t, sigma)
            medidos.append((rg, rl))
            print(f"σ={sigma:.2f} T/Tc={t:.2f}: ρ_vapor={rg:.5f} ρ_líquido={rl:.5f}")
        medidos = np.array(medidos)
        ax.plot(medidos[:, 0], temps_validas, "o", color=color, label=etiqueta)
        ax.plot(medidos[:, 1], temps_validas, "o", color=color)

    ax.plot([eos.rho_critica], [1.0], "k*", ms=10, label="punto crítico")
    ax.set_xscale("log")
    ax.set_xlabel("densidad ρ (unidades de red, escala log)")
    ax.set_ylabel("T / T_c")
    ax.set_title("Coexistencia líquido-vapor: autómata vs. termodinámica")
    ax.legend(fontsize=8, loc="lower center")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs("salidas", exist_ok=True)
    fig.savefig("salidas/diagrama_de_fases.png", dpi=140)
    print("guardado: salidas/diagrama_de_fases.png")


if __name__ == "__main__":
    main()
