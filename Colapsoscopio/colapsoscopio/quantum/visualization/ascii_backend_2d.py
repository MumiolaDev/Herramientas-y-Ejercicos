"""Backend ASCII 2D: |Psi(x,y,t)|^2 como mapa de calor de caracteres —la
escala de grises clásica ' .:-=+*#%@', no las barras verticales de
ascii_backend.py (1D), porque acá hace falta codificar dos ejes espaciales,
no uno. Cada fila impresa es un valor de x, cada columna un valor de y —
misma convención que MatplotlibAnimator2D (x vertical, y horizontal)—, así
que lo que se ve en terminal es literalmente la misma imagen que el heatmap
de matplotlib, solo que en glifos.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from colapsoscopio.quantum.simulation2d import Trajectory2D

NIVELES_ASCII = " .:-=+*#%@"

# Ver el mismo comentario en matplotlib_backend_2d.py: la densidad
# transmitida/difractada es mucho más tenue que el pico del paquete inicial
# compacto, así que sin comprimir el rango dinámico el patrón de
# interferencia no se distingue del fondo "vacío".
GAMMA_REALCE = 0.4


class AsciiAnimator2D:
    def __init__(self, trajectory: Trajectory2D, ancho: int = 64, alto: int = 28, charset: str = NIVELES_ASCII):
        self.traj = trajectory
        self.ancho = ancho
        self.alto = alto
        self.charset = charset
        self._densidad_max = max(
            (self.traj.densidad(i).max() for i in range(self.traj.n_snapshots)), default=1.0
        )
        if self._densidad_max <= 0:
            self._densidad_max = 1.0

    def _bins(self, n_puntos: int, n_celdas: int) -> np.ndarray:
        return np.linspace(0, n_puntos, n_celdas + 1, dtype=int)

    def frame(self, i: int) -> str:
        densidad = self.traj.densidad(i)  # shape (n_x, n_y)
        bordes_x = self._bins(self.traj.grid.n_x, self.alto)
        bordes_y = self._bins(self.traj.grid.n_y, self.ancho)
        niveles = len(self.charset) - 1

        filas = []
        for fi in range(self.alto):
            ax, bx = bordes_x[fi], max(bordes_x[fi + 1], bordes_x[fi] + 1)
            fila = []
            for fj in range(self.ancho):
                ay, by = bordes_y[fj], max(bordes_y[fj + 1], bordes_y[fj] + 1)
                valor = densidad[ax:bx, ay:by].max()
                intensidad = (max(valor, 0.0) / self._densidad_max) ** GAMMA_REALCE
                idx = int(round(intensidad * niveles))
                idx = min(max(idx, 0), niveles)
                fila.append(self.charset[idx])
            filas.append("".join(fila))
        # fila 0 de `densidad` es x_min: para que "arriba" del terminal sea
        # x_min igual que abajo del heatmap de matplotlib (origin="lower"),
        # se imprime de la última fila a la primera
        mapa = "\n".join(reversed(filas))

        cabecera = (
            f"t = {self.traj.tiempos[i]:8.3f}  |  norma = {self.traj.norma[i]:.6f}"
            f"  |  <E> = {self.traj.energia[i]:8.4f}"
        )
        return cabecera + "\n" + mapa

    def reproducir(self, fps: float = 12.0, paso: int = 1, limpiar_pantalla: bool = True, archivo=None) -> None:
        salida = archivo if archivo is not None else sys.stdout
        demora = 1.0 / fps if fps > 0 else 0.0
        for i in range(0, self.traj.n_snapshots, paso):
            if limpiar_pantalla:
                salida.write("\x1b[H\x1b[2J")
            salida.write(self.frame(i) + "\n")
            salida.flush()
            if demora:
                time.sleep(demora)

    def guardar_texto(self, path: str, paso: int = 1) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for i in range(0, self.traj.n_snapshots, paso):
                f.write(self.frame(i) + "\n\n")
