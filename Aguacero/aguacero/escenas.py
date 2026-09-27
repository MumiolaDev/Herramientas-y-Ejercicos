"""Escenas listas para mirar. Cada una devuelve un Mundo ya armado y,
cuando la escena tiene algo que ocurre en cada paso (un grifo abierto),
una función `cada_paso(mundo)` que el visor llama antes de avanzar.

La geometría está escrita para una caja de referencia de 200×150 y se
escala con `alto/150`: pedir 600×450 da la MISMA disposición con 3× más
pixeles por lado. Ojo con lo que eso significa físicamente: la longitud
capilar es una propiedad del fluido (~19 pixeles con los parámetros por
defecto), así que una escena más grande no es la misma escena "más
nítida" sino una caja físicamente más grande (≈ 3 cm → 8 cm de agua),
donde la gravedad y la inercia pesan más frente a la tensión superficial.
Es la forma honesta de acercarse al agua macroscópica.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from aguacero.mundo import Mundo, Parametros

Escena = tuple[Mundo, Callable[[Mundo], None] | None]


class _Lienzo:
    """Coordenadas de la caja de referencia (alto 150) → pixeles reales."""

    def __init__(self, mundo: Mundo):
        self.k = mundo.alto / 150.0
        fila, col = np.mgrid[0 : mundo.alto, 0 : mundo.ancho]
        self.fila_px, self.col_px = fila, col
        self.fila, self.col = fila / self.k, col / self.k
        self.ancho_ref = mundo.ancho / self.k
        self.alto_ref = 150.0

    def rect(self, f0, f1, c0, c1):
        return (self.fila >= f0) & (self.fila < f1) & (self.col >= c0) & (self.col < c1)

    def disco(self, fc, cc, r):
        return np.hypot(self.fila - fc, self.col - cc) < r


def gota(ancho: int = 200, alto: int = 150, motor: str = "auto", parametros: Parametros | None = None) -> Escena:
    """Una gota cae sobre una piscina: salpicadura, ondas capilares y
    (mirando la temperatura) el enfriamiento del vapor que se expande."""
    m = Mundo(ancho, alto, parametros, motor=motor)
    L = _Lienzo(m)
    m.agregar_liquido(L.fila > L.alto_ref - 25)
    m.agregar_liquido(L.disco(L.alto_ref * 0.3, L.ancho_ref / 2, 11))
    return m, None


def grifo(
    ancho: int = 200, alto: int = 150, motor: str = "auto", parametros: Parametros | None = None, u_entrada: float = 0.03
) -> Escena:
    """Una tubería vertical baja del techo; el agua entra por su extremo
    superior, sale como chorro, rebota en dos repisas y llena un
    recipiente.

    El caudal lo impone una entrada de velocidad dentro de la tubería
    (`Mundo.imponer_entrada`): una capa de 2 filas fijada cada paso al
    equilibrio de líquido con u = u_entrada. La masa NO se conserva aquí
    (entra agua); `masa_agregada` lleva la cuenta exacta."""
    m = Mundo(ancho, alto, parametros, motor=motor)
    L = _Lienzo(m)
    m.agregar_pared(L.rect(0, 22, 38, 40) | L.rect(0, 22, 47, 49))
    m.agregar_pared(L.rect(55, 58, 16, 95))
    m.agregar_pared(L.rect(95, 98, 81, 170))
    m.agregar_pared(L.rect(L.alto_ref - 40, L.alto_ref, 120, 123))
    boca = L.rect(0, 3, 40, 47)  # desde el techo: sin bolsillos de vapor atrapados sobre la entrada
    T_entrada = m.p.T_reducida

    def cada_paso(mundo: Mundo) -> None:
        mundo.imponer_entrada(boca, uy=u_entrada, T_reducida=T_entrada)

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
    L = _Lienzo(m)
    m.agregar_liquido(L.fila > L.alto_ref - 45)
    m.agregar_fuente((L.fila_px == alto - 2) & (L.col_px > ancho * 0.3) & (L.col_px < ancho * 0.7), T_placa)
    m.agregar_fuente((L.fila_px == 1) & (L.col_px > 1) & (L.col_px < ancho - 2), 0.8)
    return m, None


ESCENAS = {"gota": gota, "grifo": grifo, "tetera": tetera}
