"""Visor interactivo con pygame: cada pixel del Mundo es un pixel (escalado)
de la ventana, y puedes pintar agua, paredes, calor o frío con el mouse
mientras corre.

    python -m aguacero                 # escena "grifo"
    python -m aguacero tetera --escala 4
    python -m aguacero gota --motor numpy   # la referencia lenta

Teclas (también en pantalla con H):
    1-5      vista: agua · temperatura · presión · rapidez · vorticidad
    A P B C F   herramienta: agua · pared · borrar · calentar · enfriar
    rueda / [ ]  radio del pincel        clic der.  borrar
    ESPACIO  pausa      N  un paso       + / −  pasos por cuadro
    G  gravedad on/off   R  reiniciar    S  captura PNG    H  ayuda   ESC salir
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from aguacero.escenas import ESCENAS
from aguacero.mundo import Mundo

VISTAS = ["agua", "temperatura", "presion", "rapidez", "vorticidad"]
HERRAMIENTAS = {"a": "agua", "p": "pared", "b": "borrar", "c": "calentar", "f": "enfriar"}
T_CALENTAR = 1.15
T_ENFRIAR = 0.62

COLOR_PARED = np.array([70, 70, 78], dtype=np.uint8)
COLOR_CALOR = np.array([235, 80, 40], dtype=np.uint8)
COLOR_FRIO = np.array([60, 170, 235], dtype=np.uint8)


def _lut(nombre: str) -> np.ndarray:
    from matplotlib import colormaps

    return (colormaps[nombre](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


class Visor:
    def __init__(self, escena: str = "grifo", escala: int = 4, motor: str = "auto", pasos_por_cuadro: int = 8):
        import pygame

        self.pg = pygame
        self.nombre_escena = escena
        self.escala = escala
        self.motor = motor
        self.pasos_por_cuadro = pasos_por_cuadro
        self._cargar_escena(escena)

        self.vista = "agua"
        self.herramienta = "agua"
        self.radio = 4
        self.pausa = False
        self.ayuda = False
        self.gravedad_guardada = self.mundo.p.gravedad

        self.luts = {
            "agua": _lut("Blues"),
            "temperatura": _lut("inferno"),
            "presion": _lut("viridis"),
            "rapidez": _lut("magma"),
            "vorticidad": _lut("RdBu_r"),
        }

        pygame.init()
        self.ancho_panel = 300
        self.pantalla = pygame.display.set_mode(
            (self.mundo.ancho * escala + self.ancho_panel, max(self.mundo.alto * escala, 520))
        )
        pygame.display.set_caption("Aguacero — autómata de fluido pixel a pixel")
        self.fuente = pygame.font.SysFont("dejavusansmono,monospace", 14)
        self.fuente_grande = pygame.font.SysFont("dejavusansmono,monospace", 17, bold=True)
        self.reloj = pygame.time.Clock()
        self._pasos_por_segundo = 0.0

    # ------------------------------------------------------------ escena

    def _cargar_escena(self, nombre: str) -> None:
        self.mundo, self.cada_paso = ESCENAS[nombre](motor=self.motor)
        self.nombre_escena = nombre
        # compila el núcleo numba antes del primer cuadro (unos segundos la primera vez)
        self._avanzar(1)

    def _avanzar(self, n: int) -> None:
        for _ in range(n):
            if self.cada_paso is not None:
                self.cada_paso(self.mundo)
            self.mundo.paso(1)

    # ----------------------------------------------------------- dibujo

    def _campo_rgb(self) -> np.ndarray:
        m = self.mundo
        Tc = m.p.eos.T_critica
        fluido = m.fluido
        if self.vista == "agua":
            x = (m.rho - m.rho_vapor) / (m.rho_liquido - m.rho_vapor)
            x = 0.08 + 0.92 * np.clip(x, 0, 1)
        elif self.vista == "temperatura":
            x = (m.T / Tc - 0.55) / (1.2 - 0.55)
        elif self.vista == "presion":
            p = m.p.eos.presion(np.where(fluido, m.rho, m.rho_vapor), m.T)
            p0 = m.p.eos.presion(m.rho_vapor, m.T0)
            x = 0.5 + 0.5 * np.tanh((p - p0) / (3 * abs(p0) + 1e-12))
        elif self.vista == "rapidez":
            x = np.hypot(m.ux, m.uy) / 0.12
        else:
            vort = np.gradient(m.uy, axis=1) - np.gradient(m.ux, axis=0)
            x = 0.5 + 0.5 * np.tanh(vort / 0.01)
        idx = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        rgb = self.luts[self.vista][idx]
        rgb[m.solido] = COLOR_PARED
        calor = m.fuente & (m.T_fuente > m.T0)
        rgb[calor] = COLOR_CALOR
        rgb[m.fuente & ~calor] = COLOR_FRIO
        return rgb

    def _texto(self, s, x, y, color=(225, 225, 230), grande=False):
        f = self.fuente_grande if grande else self.fuente
        self.pantalla.blit(f.render(s, True, color), (x, y))

    def _panel(self) -> None:
        pg = self.pg
        m = self.mundo
        x0 = m.ancho * self.escala
        pg.draw.rect(self.pantalla, (22, 24, 30), (x0, 0, self.ancho_panel, self.pantalla.get_height()))
        x = x0 + 14
        y = 12
        Tc = m.p.eos.T_critica
        e = m.energias()
        liq = m.fraccion_liquido() > 0.5
        liq &= m.fluido
        T_liq = m.T[liq].mean() / Tc if liq.any() else float("nan")
        filas = [
            ("AGUACERO", None),
            (f"escena  {self.nombre_escena}   motor {m.motor}", None),
            (f"paso    {m.pasos}", None),
            (f"pasos/s {self._pasos_por_segundo:6.0f}  x{self.pasos_por_cuadro}/cuadro", None),
            ("", None),
            (f"vista   [{VISTAS.index(self.vista) + 1}] {self.vista}", (130, 200, 255)),
            (f"pincel  {self.herramienta}  r={self.radio}", (130, 200, 255)),
            (f"g       {'ON ' if m.p.gravedad else 'OFF'} {m.p.gravedad:.0e}", None),
            ("", None),
            ("masa (Σf)", (170, 170, 180)),
            (f"  total    {m.masa_total():11.4f}", None),
            (f"  agregada {m.masa_agregada:11.4f}", None),
            (f"  líquido  {liq.sum():6d} px", None),
            ("energía", (170, 170, 180)),
            (f"  cinética {e['cinetica']:11.5f}", None),
            (f"  potenc.  {e['potencial']:11.5f}", None),
            (f"  interna  {e['interna']:11.4f}", None),
            (f"  total    {e['total']:11.4f}", None),
            (f"T líquido  {T_liq:6.3f} T_c", None),
        ]
        for s, color in filas:
            if s == "AGUACERO":
                self._texto(s, x, y, (120, 190, 255), grande=True)
                y += 26
                continue
            self._texto(s, x, y, color or (225, 225, 230))
            y += 19

        # sonda: el pixel bajo el cursor
        mx, my = pg.mouse.get_pos()
        i, j = my // self.escala, mx // self.escala
        y += 8
        self._texto("pixel bajo el cursor", x, y, (170, 170, 180))
        y += 19
        if 0 <= i < m.alto and 0 <= j < m.ancho:
            if m.solido[i, j]:
                tipo = "fuente térmica" if m.fuente[i, j] else "pared"
                self._texto(f"  ({j},{i}) {tipo}", x, y)
                y += 19
                if m.fuente[i, j]:
                    self._texto(f"  T = {m.T_fuente[i, j] / Tc:.3f} T_c", x, y)
            else:
                rho, T = m.rho[i, j], m.T[i, j]
                p = float(m.p.eos.presion(rho, T))
                pc = float(m.p.eos.presion(m.p.eos.rho_critica, Tc))
                fase = "líquido" if rho > 0.5 * (m.rho_liquido + m.rho_vapor) else "vapor"
                for s in (
                    f"  ({j},{i}) {fase}",
                    f"  ρ   = {rho:.4f}",
                    f"  T   = {T / Tc:.3f} T_c",
                    f"  p   = {p / pc:+.3f} p_c",
                    f"  |u| = {np.hypot(m.ux[i, j], m.uy[i, j]):.4f}",
                ):
                    self._texto(s, x, y)
                    y += 19

        self._texto("H: ayuda", x, self.pantalla.get_height() - 24, (150, 150, 160))

    def _dibujar_ayuda(self) -> None:
        pg = self.pg
        lineas = __doc__.strip().splitlines()[9:]
        alto = 22 * len(lineas) + 20
        caja = pg.Surface((self.mundo.ancho * self.escala - 40, alto), pg.SRCALPHA)
        caja.fill((10, 12, 18, 225))
        self.pantalla.blit(caja, (20, 20))
        for k, linea in enumerate(lineas):
            self._texto(linea, 34, 30 + 22 * k)

    def dibujar(self) -> None:
        pg = self.pg
        rgb = self._campo_rgb()
        sup = pg.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
        sup = pg.transform.scale(sup, (self.mundo.ancho * self.escala, self.mundo.alto * self.escala))
        self.pantalla.fill((0, 0, 0))
        self.pantalla.blit(sup, (0, 0))
        self._panel()
        if self.ayuda:
            self._dibujar_ayuda()
        pg.display.flip()

    # ------------------------------------------------------------ pincel

    def _mascara_pincel(self, px, py):
        m = self.mundo
        i0, j0 = py // self.escala, px // self.escala
        fila, col = np.ogrid[0 : m.alto, 0 : m.ancho]
        return (fila - i0) ** 2 + (col - j0) ** 2 <= self.radio**2

    def _pintar(self, px, py, borrar=False) -> None:
        if px >= self.mundo.ancho * self.escala:
            return
        mascara = self._mascara_pincel(px, py)
        m = self.mundo
        herramienta = "borrar" if borrar else self.herramienta
        if herramienta == "agua":
            m.agregar_liquido(mascara)
        elif herramienta == "pared":
            m.agregar_pared(mascara)
        elif herramienta == "borrar":
            m.quitar_pared(mascara)
            m.quitar_liquido(mascara)
        elif herramienta == "calentar":
            m.agregar_fuente(mascara, T_CALENTAR)
        elif herramienta == "enfriar":
            m.agregar_fuente(mascara, T_ENFRIAR)

    # ------------------------------------------------------------- bucle

    def capturar(self, ruta: str) -> str:
        os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
        self.pg.image.save(self.pantalla, ruta)
        return ruta

    def _eventos(self) -> bool:
        pg = self.pg
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                return False
            if ev.type == pg.MOUSEWHEEL:
                self.radio = int(np.clip(self.radio + ev.y, 1, 30))
            if ev.type != pg.KEYDOWN:
                continue
            k = ev.unicode.lower() if ev.unicode else ""
            if ev.key == pg.K_ESCAPE:
                return False
            if k in "12345" and k:
                self.vista = VISTAS[int(k) - 1]
            elif k in HERRAMIENTAS:
                self.herramienta = HERRAMIENTAS[k]
            elif ev.key == pg.K_SPACE:
                self.pausa = not self.pausa
            elif k == "n":
                self._avanzar(1)
            elif k in ("+", "="):
                self.pasos_por_cuadro = min(self.pasos_por_cuadro * 2, 256)
            elif k == "-":
                self.pasos_por_cuadro = max(self.pasos_por_cuadro // 2, 1)
            elif k == "[":
                self.radio = max(self.radio - 1, 1)
            elif k == "]":
                self.radio = min(self.radio + 1, 30)
            elif k == "g":
                if self.mundo.p.gravedad:
                    self.gravedad_guardada, self.mundo.p.gravedad = self.mundo.p.gravedad, 0.0
                else:
                    self.mundo.p.gravedad = self.gravedad_guardada
            elif k == "r":
                self._cargar_escena(self.nombre_escena)
            elif k == "s":
                ruta = self.capturar(f"salidas/aguacero_{self.nombre_escena}_{self.mundo.pasos}.png")
                print("captura:", ruta)
            elif k == "h":
                self.ayuda = not self.ayuda
        botones = pg.mouse.get_pressed()
        if botones[0] or botones[2]:
            self._pintar(*pg.mouse.get_pos(), borrar=botones[2])
        return True

    def correr(self, max_cuadros: int | None = None) -> None:
        cuadros = 0
        t_ref, pasos_ref = time.perf_counter(), self.mundo.pasos
        while self._eventos():
            if not self.pausa:
                self._avanzar(self.pasos_por_cuadro)
            self.dibujar()
            self.reloj.tick(60)
            ahora = time.perf_counter()
            if ahora - t_ref > 0.5:
                self._pasos_por_segundo = (self.mundo.pasos - pasos_ref) / (ahora - t_ref)
                t_ref, pasos_ref = ahora, self.mundo.pasos
            cuadros += 1
            if max_cuadros is not None and cuadros >= max_cuadros:
                break
        self.pg.quit()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m aguacero", description=__doc__.split("\n\n")[0])
    ap.add_argument("escena", nargs="?", default="grifo", choices=sorted(ESCENAS))
    ap.add_argument("--escala", type=int, default=4, help="pixeles de pantalla por celda (default 4)")
    ap.add_argument("--motor", default="auto", choices=["auto", "numba", "numpy"])
    ap.add_argument("--pasos", type=int, default=8, help="pasos de simulación por cuadro")
    args = ap.parse_args(argv)
    Visor(args.escena, args.escala, args.motor, args.pasos).correr()


if __name__ == "__main__":
    main()
