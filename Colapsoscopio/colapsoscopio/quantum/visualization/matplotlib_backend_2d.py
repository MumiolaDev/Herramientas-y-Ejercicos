"""Backend matplotlib 2D: |Psi(x,y,t)|^2 como mapa de calor (imshow),
con el potencial de fondo dibujado como contorno tenue (para ver la
pantalla/rendijas sin taparlo con la densidad).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import PowerNorm

from colapsoscopio.quantum.simulation2d import Trajectory2D

# La densidad transmitida/difractada es órdenes de magnitud más tenue que el
# pico del paquete incidente todavía compacto: con una escala de color lineal
# contra el máximo global, el patrón de interferencia -la parte interesante-
# queda invisible. PowerNorm(gamma<1) comprime el rango dinámico (como el
# "stretch" no lineal habitual en imágenes astronómicas de bajo brillo) sin
# dejar de ser una función monótona de la densidad real: sigue siendo
# "más denso = más brillante" en todo momento, solo que no lineal.
GAMMA_REALCE = 0.4


class MatplotlibAnimator2D:
    def __init__(self, trajectory: Trajectory2D, potencial_xy=None):
        self.traj = trajectory
        self.potencial_xy = potencial_xy

    def _preparar_figura(self):
        grid = self.traj.grid
        densidad_max = max(self.traj.densidad(i).max() for i in range(self.traj.n_snapshots))

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        extent = [grid.y_min, grid.y_max, grid.x_min, grid.x_max]  # eje horizontal=y, vertical=x
        im = ax.imshow(
            self.traj.densidad(0),
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap="magma",
            norm=PowerNorm(gamma=GAMMA_REALCE, vmin=0, vmax=densidad_max),
        )
        if self.potencial_xy is not None:
            v = self.potencial_xy
            if v.max() > 0:
                ax.contour(grid.y, grid.x, v, levels=[v.max() * 0.5], colors="cyan", linewidths=0.8, alpha=0.6)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        titulo = ax.set_title("")
        fig.colorbar(im, ax=ax, label=r"$|\Psi(x,y,t)|^2$", shrink=0.85)
        fig.tight_layout()
        return fig, ax, im, titulo

    def animar(self, intervalo_ms: int = 40):
        fig, ax, im, titulo = self._preparar_figura()

        def actualizar(i: int):
            im.set_data(self.traj.densidad(i))
            titulo.set_text(
                f"t = {self.traj.tiempos[i]:.3f}   norma = {self.traj.norma[i]:.6f}   <E> = {self.traj.energia[i]:.4f}"
            )
            return im, titulo

        anim = FuncAnimation(fig, actualizar, frames=self.traj.n_snapshots, interval=intervalo_ms, blit=False)
        return fig, anim

    def guardar(self, path: str, fps: int = 25, dpi: int = 110) -> None:
        fig, anim = self.animar(intervalo_ms=int(1000 / fps))
        if path.endswith(".gif"):
            from matplotlib.animation import PillowWriter

            anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        else:
            anim.save(path, fps=fps, dpi=dpi)
        plt.close(fig)

    def guardar_frame(self, i: int, path: str, dpi: int = 110) -> None:
        fig, ax, im, titulo = self._preparar_figura()
        im.set_data(self.traj.densidad(i))
        titulo.set_text(
            f"t = {self.traj.tiempos[i]:.3f}   norma = {self.traj.norma[i]:.6f}   <E> = {self.traj.energia[i]:.4f}"
        )
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
