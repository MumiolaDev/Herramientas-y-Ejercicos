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
    u_antes = np.hypot(m.ux, m.uy)
    m.paso(5000)
    u = np.hypot(m.ux, m.uy)
    # En reposo quedan corrientes espurias ESTACIONARIAS que nacen en la
    # línea de contacto (artefacto conocido del pseudopotencial). Escalan
    # como ~1/ν: con τ = 0.8 el líquido queda en |u| ~ 2.5e-4, con el
    # τ = 0.52 por defecto en ~2.2e-3 (medido). Se exige que sean
    # estacionarias (no un chapoteo sin amortiguar) y acotadas.
    liquido = m.rho > 0.5 * (m.rho_liquido + m.rho_vapor)
    assert np.abs(u - u_antes)[liquido].max() < 0.05 * u[liquido].max()  # medido: ~3%
    assert u[liquido].mean() < 3e-3
    p = m.presion()[:, 20]
    rho = m.rho[:, 20]
    filas = np.arange(45, 80)  # lejos de la superficie libre y del fondo
    pendiente = np.polyfit(filas, p[filas], 1)[0]  # fila crece hacia abajo
    assert pendiente == pytest.approx(rho[filas].mean() * g, rel=0.03)
