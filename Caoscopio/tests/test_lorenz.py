"""Ver el docstring de systems/lorenz.py: dos validaciones independientes,
ninguna apoyada en observar hacia dónde "parece" que converge la
trayectoria.
"""

from __future__ import annotations

import numpy as np
import pytest

from caoscopio import Lorenz, Simulation, SimulationConfig
from caoscopio.core.integrator import RK4Integrator


def test_puntos_fijos_son_equilibrios_exactos():
    sistema = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    for punto_fijo in sistema.puntos_fijos():
        derivadas = sistema.derivadas(0.0, punto_fijo)
        assert derivadas == pytest.approx(np.zeros(3), abs=1e-10)


def test_tasa_de_contraccion_de_volumen_coincide_con_la_divergencia_teorica():
    sistema = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    integrador = RK4Integrator(sistema)

    eps = 1e-5
    centro = np.array([1.0, 1.0, 20.0])
    puntos = np.array([centro] + [centro + eps * np.eye(3)[i] for i in range(3)])

    def volumen_tetraedro(p):
        v1, v2, v3 = p[1] - p[0], p[2] - p[0], p[3] - p[0]
        return abs(np.dot(v1, np.cross(v2, v3))) / 6

    vol0 = volumen_tetraedro(puntos)

    dt, n_steps = 1e-3, 500
    estados = puntos.copy()
    for paso in range(n_steps):
        t = paso * dt
        estados = np.array([integrador.paso(t, p, dt) for p in estados])

    vol_final = volumen_tetraedro(estados)
    tasa_medida = np.log(vol_final / vol0) / (n_steps * dt)

    assert tasa_medida == pytest.approx(sistema.divergencia, rel=0.02)


def test_atractor_permanece_acotado_pese_a_ser_caotico():
    """Disipativo + caótico no es contradictorio: el volumen se contrae
    (test anterior) y aun así la trayectoria no converge a un punto — pero
    tampoco escapa a infinito. Chequeo grueso de sanidad."""
    sistema = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    config = SimulationConfig(
        sistema=sistema, estado_inicial=np.array([1.0, 1.0, 1.0]), dt=1e-3, n_steps=20000, guardar_cada=20
    )
    traj = Simulation(config).run()
    assert np.all(np.isfinite(traj.estados))
    assert np.max(np.abs(traj.estados)) < 100  # el atractor clásico vive en un rango de orden 10-50
