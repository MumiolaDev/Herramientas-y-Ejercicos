"""El péndulo forzado y amortiguado no conserva energía, así que la
validación no puede apoyarse en eso (ver systems/driven_pendulum.py). Tres
chequeos independientes en su lugar:

1. Sin forzado (A=0) y amplitud pequeña, debe reducirse al oscilador
   armónico amortiguado clásico — solución cerrada conocida, la misma
   idea de "comparar contra un caso soluble" que se usa en todo el
   proyecto.
2. Con A=0.9 (ciclo límite, verificado numéricamente en esta conversación
   antes de escribir el test), varias condiciones iniciales bien distintas
   deben converger al *mismo* punto en la sección de Poincaré
   estroboscópica — la firma de que es un atractor genuino, no un
   artefacto de la condición inicial.
3. Con A=1.15 (caótico, mismo verificado), la sección de Poincaré no debe
   colapsar a un punto — desviación estándar apreciable, contraste directo
   con el caso anterior.
"""

from __future__ import annotations

import numpy as np
import pytest

from caoscopio import DrivenDampedPendulum, Simulation, SimulationConfig


def test_decaimiento_subamortiguado_sin_forzado_coincide_con_solucion_analitica():
    g_l, b, theta0 = 1.0, 0.5, 0.05
    sistema = DrivenDampedPendulum(l=1.0, g=g_l, b=b, A=0.0, omega_d=2 / 3)
    config = SimulationConfig(
        sistema=sistema, estado_inicial=np.array([theta0, 0.0]), dt=1e-3, n_steps=15000, guardar_cada=50
    )
    traj = Simulation(config).run()

    omega1 = np.sqrt(g_l - (b / 2) ** 2)
    t = traj.tiempos
    theta_analitico = theta0 * np.exp(-b * t / 2) * (
        np.cos(omega1 * t) + (b / (2 * omega1)) * np.sin(omega1 * t)
    )
    assert traj.estados[:, 0] == pytest.approx(theta_analitico, abs=2e-5)


def _punto_poincare(sistema, estado_inicial, n_periodos=300):
    N_periodo = 300
    dt = sistema.periodo_forzado / N_periodo
    config = SimulationConfig(sistema, np.array(estado_inicial), dt, N_periodo * n_periodos, N_periodo)
    traj = Simulation(config).run()
    return traj.estados[100:]  # descarta el transiente (primeros 100 períodos)


def test_atractor_periodico_es_independiente_de_la_condicion_inicial():
    sistema = DrivenDampedPendulum(A=0.9)
    condiciones = [[0.2, 0.0], [2.5, 1.0], [-1.5, -2.0], [3.0, 0.5]]
    puntos_finales = [_punto_poincare(sistema, ic, n_periodos=250)[-1] for ic in condiciones]

    referencia = puntos_finales[0]
    for punto in puntos_finales[1:]:
        # comparar theta mod 2*pi (el ángulo es cíclico)
        dtheta = (punto[0] - referencia[0] + np.pi) % (2 * np.pi) - np.pi
        assert abs(dtheta) < 1e-3
        assert punto[1] == pytest.approx(referencia[1], abs=1e-3)


def test_atractor_caotico_no_colapsa_a_un_punto():
    sistema = DrivenDampedPendulum(A=1.15)
    puntos = _punto_poincare(sistema, [0.2, 0.0], n_periodos=400)
    theta_mod = (puntos[:, 0] + np.pi) % (2 * np.pi) - np.pi
    assert theta_mod.std() > 0.5, "se espera dispersión apreciable en el régimen caótico"
