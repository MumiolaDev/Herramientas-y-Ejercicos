"""El split-operator 2D es la misma idea que el 1D (ver test_conservacion.py)
sobre FFT2 en vez de FFT/DST: la norma debe conservarse igual de bien. El
caso de uso que de verdad vale la pena fijar como regresión es más
específico: que la doble rendija produzca un patrón de *interferencia* (un
perfil no monótono, con un mínimo entre dos máximos) y no dos manchas
separadas — la firma de que el split-operator 2D está resolviendo
difracción de verdad, no solo transportando el paquete.
"""

from __future__ import annotations

import numpy as np
import pytest

from colapsoscopio import Grid2D, Simulation2D, SimulationConfig2D
from colapsoscopio.quantum.initial_conditions import GaussianPacket2D
from colapsoscopio.quantum.potentials import DoubleSlit


def test_norma_y_energia_se_conservan_en_2d():
    grid = Grid2D(x_min=-15.0, x_max=15.0, n_x=128, y_min=-15.0, y_max=15.0, n_y=128)
    potential = DoubleSlit(v0=40.0, x_pantalla=0.0, grosor=0.5, separacion=3.0, ancho_rendija=0.8)
    ic = GaussianPacket2D(x0=-6.0, y0=0.0, sigma_x=1.2, sigma_y=2.0, kx0=3.0, ky0=0.0)
    config = SimulationConfig2D(
        grid=grid, potential=potential, initial_condition=ic, dt=2e-3, n_steps=800, guardar_cada=80
    )
    traj = Simulation2D(config).run()

    assert traj.norma == pytest.approx(1.0, abs=1e-8)
    drift_relativo = abs(traj.energia[-1] - traj.energia[0]) / abs(traj.energia[0])
    assert drift_relativo < 1e-4


def test_doble_rendija_produce_interferencia_no_dos_manchas():
    grid = Grid2D(x_min=-20.0, x_max=20.0, n_x=256, y_min=-15.0, y_max=15.0, n_y=256)
    potential = DoubleSlit(v0=40.0, x_pantalla=0.0, grosor=0.5, separacion=3.0, ancho_rendija=0.8)
    ic = GaussianPacket2D(x0=-8.0, y0=0.0, sigma_x=1.5, sigma_y=3.0, kx0=3.0, ky0=0.0)
    config = SimulationConfig2D(
        grid=grid, potential=potential, initial_condition=ic, dt=2e-3, n_steps=3000, guardar_cada=3000
    )
    traj = Simulation2D(config).run()

    x = grid.x
    idx_x = int(np.argmin(np.abs(x - 6.0)))  # un corte bastante después de la pantalla
    perfil = traj.densidad(-1)[idx_x, :]

    centro = len(perfil) // 2
    # el máximo central de interferencia debe superar a sus vecinos inmediatos
    assert perfil[centro] > perfil[centro - 5]
    assert perfil[centro] > perfil[centro + 5]

    # recorriendo desde el centro hacia +y, el perfil debe bajar (mínimo de
    # interferencia) y luego volver a subir (máximo secundario) — esa
    # no-monotonía es la firma de interferencia; un perfil de "una sola
    # mancha" difractada sería monótonamente decreciente hacia afuera.
    mitad_derecha = perfil[centro:]
    idx_minimo = int(np.argmin(mitad_derecha[:len(mitad_derecha) // 2]))
    valor_minimo = mitad_derecha[idx_minimo]
    valor_maximo_despues = mitad_derecha[idx_minimo:].max()
    assert idx_minimo > 2, "el mínimo de interferencia no debería estar pegado al centro"
    assert valor_maximo_despues > valor_minimo * 1.5, "se espera un máximo secundario claro tras el mínimo"
