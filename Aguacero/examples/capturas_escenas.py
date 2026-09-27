"""Corre las tres escenas sin ventana y guarda una tira de cuadros de
densidad y temperatura de cada una. Útil para ver qué hace cada escena
sin abrir pygame, o en una máquina sin pantalla.

Corre:
    python examples/capturas_escenas.py

Genera:
    salidas/escena_gota.png, salidas/escena_grifo.png, salidas/escena_tetera.png
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aguacero.escenas import ESCENAS

TIEMPOS = {"gota": [0, 800, 1600, 3000], "grifo": [1500, 5000, 10000, 16000], "tetera": [4000, 12000, 20000, 25000]}


def main() -> None:
    os.makedirs("salidas", exist_ok=True)
    for nombre, tiempos in TIEMPOS.items():
        m, cada_paso = ESCENAS[nombre]()
        Tc = m.p.eos.T_critica
        fig, ejes = plt.subplots(2, len(tiempos), figsize=(3.6 * len(tiempos), 5.4))
        t0 = time.perf_counter()
        for k, t in enumerate(tiempos):
            while m.pasos < t:
                if cada_paso is not None:
                    cada_paso(m)
                m.paso(1)
            ejes[0, k].imshow(np.where(m.solido, np.nan, m.fraccion_liquido()), cmap="Blues", vmin=-0.1, vmax=1)
            ejes[0, k].set_title(f"{nombre}  paso {m.pasos}", fontsize=9)
            im = ejes[1, k].imshow(np.where(m.solido, np.nan, m.T / Tc), cmap="inferno", vmin=0.6, vmax=1.15)
            for ax in ejes[:, k]:
                ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ejes[1, :].tolist(), label="T / T_c", shrink=0.8)
        fig.savefig(f"salidas/escena_{nombre}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"{nombre}: {m.pasos} pasos en {time.perf_counter() - t0:.1f} s (motor {m.motor}) → salidas/escena_{nombre}.png")


if __name__ == "__main__":
    main()
