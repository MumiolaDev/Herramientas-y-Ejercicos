"""Backend de visualización con matplotlib: |Psi(x,t)|^2 relleno, Re/Im Psi
como líneas, y V(x) de fondo (reescalado para que quepa en el mismo panel).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend sin display: sirve tanto en escritorio como headless

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from colapsoscopio.quantum.simulation import Trajectory


class MatplotlibAnimator:
    def __init__(self, trajectory: Trajectory, potencial_x=None):
        self.traj = trajectory
        self.potencial_x = potencial_x  # V(x) evaluado en grid.x, opcional (solo para dibujarlo de fondo)

    def _preparar_figura(self):
        grid = self.traj.grid
        densidad_max = self.traj.densidad(0).max() if self.traj.n_snapshots else 1.0
        for i in range(self.traj.n_snapshots):
            densidad_max = max(densidad_max, self.traj.densidad(i).max())

        fig, (ax_densidad, ax_partes) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        if self.potencial_x is not None:
            v = self.potencial_x
            v_escalado = v / (v.max() + 1e-12) * densidad_max * 0.9 if v.max() > 0 else v
            ax_densidad.plot(grid.x, v_escalado, color="gray", lw=1, ls="--", label="V(x) (esc.)")

        (linea_densidad,) = ax_densidad.plot([], [], color="C0", lw=2)
        relleno = ax_densidad.fill_between(grid.x, 0, self.traj.densidad(0), color="C0", alpha=0.3)
        ax_densidad.set_ylim(0, densidad_max * 1.15)
        ax_densidad.set_ylabel(r"$|\Psi(x,t)|^2$")
        ax_densidad.legend(loc="upper right", fontsize=8)
        titulo = ax_densidad.set_title("")

        (linea_re,) = ax_partes.plot([], [], color="C1", lw=1, label=r"Re $\Psi$")
        (linea_im,) = ax_partes.plot([], [], color="C2", lw=1, label=r"Im $\Psi$")
        amplitud_max = max(
            1e-12,
            max(
                max(abs(self.traj.estados[i].real).max(), abs(self.traj.estados[i].imag).max())
                for i in range(self.traj.n_snapshots)
            ),
        )
        ax_partes.set_ylim(-amplitud_max * 1.15, amplitud_max * 1.15)
        ax_partes.set_xlabel("x")
        ax_partes.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        return fig, ax_densidad, linea_densidad, titulo, ax_partes, linea_re, linea_im

    def animar(self, intervalo_ms: int = 40):
        """Construye la FuncAnimation (no la muestra ni la guarda todavía)."""
        fig, ax_densidad, linea_densidad, titulo, ax_partes, linea_re, linea_im = (
            self._preparar_figura()
        )
        grid = self.traj.grid

        def actualizar(i: int):
            linea_densidad.set_data(grid.x, self.traj.densidad(i))
            linea_re.set_data(grid.x, self.traj.estados[i].real)
            linea_im.set_data(grid.x, self.traj.estados[i].imag)
            titulo.set_text(
                f"t = {self.traj.tiempos[i]:.3f}   "
                f"norma = {self.traj.norma[i]:.6f}   "
                f"<E> = {self.traj.energia[i]:.4f}"
            )
            return linea_densidad, linea_re, linea_im, titulo

        anim = FuncAnimation(
            fig, actualizar, frames=self.traj.n_snapshots, interval=intervalo_ms, blit=False
        )
        return fig, anim

    def guardar(self, path: str, fps: int = 25, dpi: int = 100) -> None:
        """Guarda la animación a disco. Usa PillowWriter para .gif (no
        requiere ffmpeg instalado); para .mp4 se necesita ffmpeg disponible
        en el sistema.
        """
        fig, anim = self.animar(intervalo_ms=int(1000 / fps))
        if path.endswith(".gif"):
            from matplotlib.animation import PillowWriter

            anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        else:
            anim.save(path, fps=fps, dpi=dpi)
        plt.close(fig)

    def guardar_frame(self, i: int, path: str, dpi: int = 100) -> None:
        """Guarda un único frame como imagen estática (útil para revisar
        rápido sin generar la animación completa)."""
        fig, ax_densidad, linea_densidad, titulo, ax_partes, linea_re, linea_im = (
            self._preparar_figura()
        )
        grid = self.traj.grid
        linea_densidad.set_data(grid.x, self.traj.densidad(i))
        linea_re.set_data(grid.x, self.traj.estados[i].real)
        linea_im.set_data(grid.x, self.traj.estados[i].imag)
        titulo.set_text(
            f"t = {self.traj.tiempos[i]:.3f}   "
            f"norma = {self.traj.norma[i]:.6f}   "
            f"<E> = {self.traj.energia[i]:.4f}"
        )
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
