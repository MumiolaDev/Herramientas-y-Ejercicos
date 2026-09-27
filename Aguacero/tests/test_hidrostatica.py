"""Piscina en reposo bajo gravedad: dentro del líquido la presión debe
crecer con la profundidad como dp/dz = ρ_l g (equilibrio hidrostático).
Prueba que la gravedad entra como fuerza de cuerpo bien acoplada a la
ecuación de estado."""

from __future__ import annotations

import numpy as np
import pytest

from aguacero.mundo import Mundo, Parametros


def test_gradiente_hidrostatico():
    g = 5e-5
    m = Mundo(40, 90, Parametros(gravedad=g, isotermico=True))
    fila, _ = np.mgrid[0:90, 0:40]
    m.agregar_liquido(fila > 30)
    m.paso(15000)
    # en reposo el LÍQUIDO queda quieto; en el vapor junto a la línea de
    # contacto persiste una corriente espuria |u| ~ 0.02 (artefacto
    # conocido del pseudopotencial, ver README), por eso se mide en el líquido
    liquido = m.rho > 0.5 * (m.rho_liquido + m.rho_vapor)
    assert np.hypot(m.ux, m.uy)[liquido].mean() < 1e-3
    p = m.presion()[:, 20]
    rho = m.rho[:, 20]
    filas = np.arange(45, 80)  # lejos de la superficie libre y del fondo
    pendiente = np.polyfit(filas, p[filas], 1)[0]  # fila crece hacia abajo
    assert pendiente == pytest.approx(rho[filas].mean() * g, rel=0.03)
