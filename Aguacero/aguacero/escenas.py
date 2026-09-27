"""Escenas listas para mirar. Cada una devuelve un Mundo ya armado y,
cuando la escena tiene algo que ocurre en cada paso (un grifo abierto),
una función `cada_paso(mundo)` que el visor llama antes de avanzar."""

from __future__ import annotations

from typing import Callable

import numpy as np

from aguacero.mundo import Mundo, Parametros

Escena = tuple[Mundo, Callable[[Mundo], None] | None]


def _grilla(mundo: Mundo):
    return np.mgrid[0 : mundo.alto, 0 : mundo.ancho]


def gota(ancho: int = 200, alto: int = 150, motor: str = "auto", parametros: Parametros | None = None) -> Escena:
    """Una gota cae sobre una piscina: salpicadura, ondas capilares y
    (mirando la temperatura) el enfriamiento del vapor que se expande."""
    m = Mundo(ancho, alto, parametros, motor=motor)
    fila, col = _grilla(m)
    m.agregar_liquido(fila > alto - 25)
    m.agregar_liquido(np.hypot(col - ancho / 2, fila - alto * 0.3) < 11)
    return m, None


def grifo(ancho: int = 200, alto: int = 150, motor: str = "auto", parametros: Parametros | None = None) -> Escena:
    """Un chorro entra por arriba a la izquierda, rebota en dos repisas y
    llena un recipiente. La masa NO se conserva aquí (entra agua); el
    contador `masa_agregada` del Mundo lleva la cuenta exacta de cuánto."""
    m = Mundo(ancho, alto, parametros, motor=motor)
    fila, col = _grilla(m)
    m.agregar_pared((fila >= 55) & (fila < 58) & (col > 15) & (col < 95))
    m.agregar_pared((fila >= 95) & (fila < 98) & (col > 80) & (col < 170))
    m.agregar_pared((fila > alto - 40) & (col >= 120) & (col < 123))
    boca = (fila >= 12) & (fila < 17) & (col >= 40) & (col < 47)

    def cada_paso(mundo: Mundo) -> None:
        if mundo.pasos % 3 == 0:
            mundo.agregar_liquido(boca, uy=0.06)

    return m, cada_paso


def tetera(
    ancho: int = 200, alto: int = 150, motor: str = "auto", T_placa: float = 1.1, parametros: Parametros | None = None
) -> Escena:
    """Una piscina sobre una placa caliente (T_placa en unidades de T_c) y
    un techo frío. El líquido se calienta desde abajo, se expande, y cerca
    de la placa la EOS deja de admitir líquido: aparece vapor que sube
    como burbujas, se condensa arriba y vuelve a caer. El calor latente no
    está puesto a mano — sale del trabajo de compresión (ver termico.py)."""
    m = Mundo(ancho, alto, parametros, motor=motor)
    fila, col = _grilla(m)
    m.agregar_liquido(fila > alto - 45)
    m.agregar_fuente((fila == alto - 2) & (col > ancho * 0.3) & (col < ancho * 0.7), T_placa)
    m.agregar_fuente((fila == 1) & (col > 1) & (col < ancho - 2), 0.8)
    return m, None


ESCENAS = {"gota": gota, "grifo": grifo, "tetera": tetera}
