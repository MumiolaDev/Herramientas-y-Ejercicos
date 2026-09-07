"""Backend de visualización puramente en terminal, sin dependencias gráficas.

Cada frame se reduce a UNA línea de "sparkline" (bloques Unicode de altura
creciente ▁▂▃▄▅▆▇█, o el charset ASCII puro ' .:-=+*#%@' si el terminal no
soporta UTF-8) que representa |Psi(x,t)|^2 a lo ancho de la malla, más una
cabecera con t, norma y <E>. Reproducir la trayectoria es entonces imprimir
frame tras frame limpiando pantalla entre uno y otro — un "GIF" que vive en
cualquier terminal, sin necesidad de matplotlib ni de un entorno gráfico.
Útil también como sanity-check rápido en un sandbox sin display.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from colapsoscopio.quantum.simulation import Trajectory

BLOQUES_UNICODE = " ▁▂▃▄▅▆▇█"
BLOQUES_ASCII = " .:-=+*#%@"

_LIMPIAR_PANTALLA = "\x1b[H\x1b[2J"


class AsciiAnimator:
    def __init__(
        self,
        trajectory: Trajectory,
        ancho: int = 78,
        charset: str = BLOQUES_UNICODE,
    ):
        self.traj = trajectory
        self.ancho = ancho
        self.charset = charset
        # normalizamos todos los frames contra el máximo global, no el de
        # cada frame por separado: así se ve la dispersión/decaimiento real
        # del paquete en vez de un reescalado que lo ocultaría
        self._densidad_max = max(
            (self.traj.densidad(i).max() for i in range(self.traj.n_snapshots)), default=1.0
        )
        if self._densidad_max <= 0:
            self._densidad_max = 1.0

    def _bins(self) -> np.ndarray:
        """Índices de borde para reagrupar n_points en `ancho` columnas."""
        return np.linspace(0, self.traj.grid.n_points, self.ancho + 1, dtype=int)

    def frame(self, i: int) -> str:
        densidad = self.traj.densidad(i)
        bordes = self._bins()
        niveles = len(self.charset) - 1
        columnas = []
        for j in range(self.ancho):
            a, b = bordes[j], bordes[j + 1]
            b = max(b, a + 1)  # evita bins vacíos si ancho > n_points
            valor = densidad[a:b].max()
            idx = int(round((valor / self._densidad_max) * niveles))
            idx = min(max(idx, 0), niveles)
            columnas.append(self.charset[idx])
        barra = "".join(columnas)
        cabecera = (
            f"t = {self.traj.tiempos[i]:8.3f}  |  norma = {self.traj.norma[i]:.6f}"
            f"  |  <E> = {self.traj.energia[i]:8.4f}  |  <x> = {self.traj.x_esperado[i]:7.3f}"
        )
        return cabecera + "\n" + barra

    def reproducir(self, fps: float = 15.0, paso: int = 1, limpiar_pantalla: bool = True, archivo=None) -> None:
        """Imprime la trayectoria frame a frame en `archivo` (por defecto
        sys.stdout), a `fps` cuadros por segundo, tomando 1 de cada `paso`
        snapshots guardados.
        """
        salida = archivo if archivo is not None else sys.stdout
        demora = 1.0 / fps if fps > 0 else 0.0
        for i in range(0, self.traj.n_snapshots, paso):
            if limpiar_pantalla:
                salida.write(_LIMPIAR_PANTALLA)
            salida.write(self.frame(i) + "\n")
            salida.flush()
            if demora:
                time.sleep(demora)

    def guardar_texto(self, path: str, paso: int = 1) -> None:
        """Guarda todos los frames (separados por una línea en blanco) en un
        archivo de texto plano, para inspeccionar sin necesidad de una
        terminal interactiva (por ejemplo, en CI o en este mismo sandbox)."""
        with open(path, "w", encoding="utf-8") as f:
            for i in range(0, self.traj.n_snapshots, paso):
                f.write(self.frame(i) + "\n\n")
