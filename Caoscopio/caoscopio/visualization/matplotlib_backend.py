"""Backend matplotlib: dos paneles lado a lado — el péndulo en el espacio
real (con una estela detrás de la última masa) y su retrato de fases
(θ, ω del grado de libertad que se pida), ambos animados cuadro a cuadro.
Acepta una segunda trayectoria "gemela" opcional (mismo sistema, condición
inicial ligerísimamente distinta) para las demos de sensibilidad: se
dibuja superpuesta en un color de contraste, en ambos paneles a la vez.
"""

from __future__ import annotations

from collections import deque

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from caoscopio.core.trajectory import Trajectory
from caoscopio.systems.double_pendulum import DoublePendulum


class PendulumAnimator:
    def __init__(
        self,
        sistema,
        trayectoria: Trajectory,
        trayectoria_gemela: Trajectory | None = None,
        indices_fase: tuple[int, int] | None = None,
        longitud_estela: int = 300,
    ):
        self.sistema = sistema
        self.traj = trayectoria
        self.gemela = trayectoria_gemela
        self.es_doble = isinstance(sistema, DoublePendulum)
        # por defecto: retrato de fase del último péndulo (el que más se
        # mueve en el régimen caótico) — (θ2,ω2) en el doble, (θ,ω) en el simple
        self.idx_theta, self.idx_omega = indices_fase or ((2, 3) if self.es_doble else (0, 1))
        self.longitud_estela = longitud_estela

    def _posiciones(self, y):
        if self.es_doble:
            return self.sistema.posiciones(y)  # ((x1,y1),(x2,y2))
        return (self.sistema.posicion(y),)  # ((x,y),)

    def _preparar_figura(self):
        fig, (ax_fis, ax_fase) = plt.subplots(1, 2, figsize=(11, 5.2))

        alcance = self.sistema.l1 + self.sistema.l2 if self.es_doble else self.sistema.l
        margen = alcance * 1.15
        ax_fis.set_xlim(-margen, margen)
        ax_fis.set_ylim(-margen, margen)
        ax_fis.set_aspect("equal")
        ax_fis.set_xticks([]); ax_fis.set_yticks([])
        for spine in ax_fis.spines.values():
            spine.set_visible(False)
        ax_fis.axhline(0, color="#00000010", lw=0.6)

        (varillas,) = ax_fis.plot([], [], "-", color="#52514e", lw=1.6, zorder=2)
        (masas,) = ax_fis.plot([], [], "o", color="#2a78d6", ms=9, zorder=3)
        (estela,) = ax_fis.plot([], [], "-", color="#2a78d6", lw=1, alpha=0.55, zorder=1)
        elementos_fis = {"varillas": varillas, "masas": masas, "estela": estela}

        varillas_g = masas_g = estela_g = None
        if self.gemela is not None:
            (varillas_g,) = ax_fis.plot([], [], "-", color="#eb683480", lw=1.6, zorder=2)
            (masas_g,) = ax_fis.plot([], [], "o", color="#eb6834", ms=8, zorder=3)
            (estela_g,) = ax_fis.plot([], [], "-", color="#eb6834", lw=1, alpha=0.55, zorder=1)
        elementos_fis.update({"varillas_g": varillas_g, "masas_g": masas_g, "estela_g": estela_g})

        titulo = ax_fis.set_title("", fontsize=10, family="monospace")

        theta_all = self.traj.estados[:, self.idx_theta]
        omega_all = self.traj.estados[:, self.idx_omega]
        if self.gemela is not None:
            theta_all = np.concatenate([theta_all, self.gemela.estados[:, self.idx_theta]])
            omega_all = np.concatenate([omega_all, self.gemela.estados[:, self.idx_omega]])
        margen_t = 0.08 * (theta_all.max() - theta_all.min() + 1e-9)
        margen_w = 0.08 * (omega_all.max() - omega_all.min() + 1e-9)
        ax_fase.set_xlim(theta_all.min() - margen_t, theta_all.max() + margen_t)
        ax_fase.set_ylim(omega_all.min() - margen_w, omega_all.max() + margen_w)
        etiqueta = f"θ{2 if (self.es_doble and self.idx_theta==2) else (1 if self.es_doble else '')}"
        ax_fase.set_xlabel(f"{etiqueta} (rad)")
        ax_fase.set_ylabel(f"ω{etiqueta[1:]} (rad/s)" if etiqueta != "θ" else "ω (rad/s)")
        ax_fase.axhline(0, color="#00000015", lw=0.6)
        ax_fase.axvline(0, color="#00000015", lw=0.6)

        (traza,) = ax_fase.plot([], [], "-", color="#2a78d6", lw=0.8, alpha=0.75)
        (punto,) = ax_fase.plot([], [], "o", color="#2a78d6", ms=6)
        elementos_fase = {"traza": traza, "punto": punto}
        traza_g = punto_g = None
        if self.gemela is not None:
            (traza_g,) = ax_fase.plot([], [], "-", color="#eb6834", lw=0.8, alpha=0.75)
            (punto_g,) = ax_fase.plot([], [], "o", color="#eb6834", ms=5)
        elementos_fase.update({"traza_g": traza_g, "punto_g": punto_g})

        fig.tight_layout()
        return fig, elementos_fis, elementos_fase, titulo

    def _actualizar_frame(self, i, elementos_fis, elementos_fase, titulo, estelas):
        def _dibujar(traj, prefijo, estela_deque):
            y = traj.estados[i]
            puntos = self._posiciones(y)
            xs = [0.0] + [p[0] for p in puntos]
            ys = [0.0] + [p[1] for p in puntos]
            elementos_fis[f"varillas{prefijo}"].set_data(xs, ys)
            elementos_fis[f"masas{prefijo}"].set_data(xs[1:], ys[1:])
            estela_deque.append(puntos[-1])
            elementos_fis[f"estela{prefijo}"].set_data(
                [p[0] for p in estela_deque], [p[1] for p in estela_deque]
            )
            th = traj.estados[: i + 1, self.idx_theta]
            om = traj.estados[: i + 1, self.idx_omega]
            elementos_fase[f"traza{prefijo}"].set_data(th, om)
            elementos_fase[f"punto{prefijo}"].set_data([th[-1]], [om[-1]])

        _dibujar(self.traj, "", estelas[0])
        if self.gemela is not None:
            _dibujar(self.gemela, "_g", estelas[1])

        titulo.set_text(f"t = {self.traj.tiempos[i]:.2f} s")
        artistas = list(elementos_fis.values()) + list(elementos_fase.values()) + [titulo]
        return [a for a in artistas if a is not None]

    def animar(self, intervalo_ms: int = 30):
        fig, elementos_fis, elementos_fase, titulo = self._preparar_figura()
        estelas = (deque(maxlen=self.longitud_estela), deque(maxlen=self.longitud_estela))

        def actualizar(i):
            return self._actualizar_frame(i, elementos_fis, elementos_fase, titulo, estelas)

        anim = FuncAnimation(
            fig, actualizar, frames=self.traj.n_snapshots, interval=intervalo_ms, blit=False
        )
        return fig, anim

    def guardar(self, path: str, fps: int = 30, dpi: int = 100) -> None:
        fig, anim = self.animar(intervalo_ms=int(1000 / fps))
        if path.endswith(".gif"):
            from matplotlib.animation import PillowWriter

            anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        else:
            anim.save(path, fps=fps, dpi=dpi)
        plt.close(fig)

    def guardar_frame(self, i: int, path: str, dpi: int = 100) -> None:
        fig, elementos_fis, elementos_fase, titulo = self._preparar_figura()
        estelas = (deque(maxlen=self.longitud_estela), deque(maxlen=self.longitud_estela))
        for j in range(max(0, i - self.longitud_estela), i + 1):
            self._actualizar_frame(j, elementos_fis, elementos_fase, titulo, estelas)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
