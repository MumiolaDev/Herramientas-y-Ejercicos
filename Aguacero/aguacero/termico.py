"""Ecuación de energía: temperatura como campo propio, acoplado al fluido.

Modelo "híbrido" de Li, Kang, Francois, He & Luo (2015): el flujo lo
resuelve lattice Boltzmann y la temperatura, diferencias finitas sobre la
misma grilla (mismo autómata: cada pixel mira sólo a sus vecinos),

    ∂T/∂t = −u·∇T + (1/ρc_v) ∇·(λ∇T) − (T/ρc_v) (∂p_EOS/∂T)_ρ ∇·u.

El último término es el trabajo de compresión y es el que hace a esto
termodinámica y no un colorante pasivo: donde el líquido se evapora el
fluido se expande (∇·u > 0) y se ENFRÍA — el calor latente no se pone a
mano, sale de la EOS. Con λ = ρ c_v χ (difusividad χ uniforme, la misma
elección de Li et al.) el término difusivo queda χ ∇·(ρ∇T)/ρ.

Lo que este modelo NO incluye, dicho de frente: el calentamiento viscoso
(la energía cinética disipada no vuelve a T) y la contribución de la
energía de gradiente de la interfaz. Por eso el balance de energía total
en aguacero/mundo.py se reporta, no se asume.
"""

from __future__ import annotations

import numpy as np

from aguacero.eos import CarnahanStarling
from aguacero.lattice import gradiente_isotropico


def _flujo_difusivo(T, rho, fluido, conductor):
    """∇·(ρ∇T)/ρ en forma de volúmenes finitos, con dos decisiones que
    importan:

    - Conductancia de cara = media ARMÓNICA de ρ a ambos lados. Es la
      regla correcta para conductividad discontinua (resistencias en
      serie, Patankar 1980) y además es la que mantiene estable el
      esquema explícito: en una cara vapor|líquido (ρ 0.008 | 0.30) la
      media aritmética daría una difusividad efectiva ~20χ del lado del
      vapor y el paso explota; la armónica acota ese factor por 2.
    - Paredes adiabáticas: la cara sólo conduce si ambas celdas son
      conductoras (fluido o fuente). Neumann exacto, sin celdas fantasma.
      Frente a una fuente térmica la conductancia es la del fluido.
    """
    total = np.zeros_like(T)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        T_v = np.roll(T, (-dy, -dx), axis=(0, 1))
        rho_v = np.roll(rho, (-dy, -dx), axis=(0, 1))
        cond_v = np.roll(conductor, (-dy, -dx), axis=(0, 1))
        fluido_v = np.roll(fluido, (-dy, -dx), axis=(0, 1))
        rho_cara = np.where(fluido_v, 2.0 * rho * rho_v / (rho + rho_v), rho)
        total += np.where(conductor & cond_v, rho_cara * (T_v - T), 0.0)
    return total / rho


def _adveccion_upwind(T, ux, uy):
    """−u·∇T con diferencias contra el viento. La centrada es más precisa
    pero con Heun (RK2) su espectro cae sobre el eje imaginario, fuera de
    la región de estabilidad, y en los transitorios (|u| ~ 0.2 al caer una
    gota) el número de Péclet de celda supera 2. La difusión numérica que
    agrega upwind, ~|u|/2, es comparable a χ: se acepta a cambio de no
    tener que vigilar el paso."""
    dTx_menos = T - np.roll(T, 1, axis=1)
    dTx_mas = np.roll(T, -1, axis=1) - T
    dTy_menos = T - np.roll(T, 1, axis=0)
    dTy_mas = np.roll(T, -1, axis=0) - T
    return -(
        np.maximum(ux, 0.0) * dTx_menos
        + np.minimum(ux, 0.0) * dTx_mas
        + np.maximum(uy, 0.0) * dTy_menos
        + np.minimum(uy, 0.0) * dTy_mas
    )


def lado_derecho(T, rho, ux, uy, div_u, fluido, conductor, eos: CarnahanStarling, chi):
    adveccion = _adveccion_upwind(T, ux, uy)
    difusion = chi * _flujo_difusivo(T, rho, fluido, conductor)
    compresion = -T * eos.dp_dT(rho) / (rho * eos.cv) * div_u
    return np.where(fluido, adveccion + difusion + compresion, 0.0)


def rellenar_paredes(T, fluido, fuente):
    """En paredes adiabáticas T no tiene significado físico, pero la
    advección upwind la lee (junto a la pared u ≈ 0, así que pesa poco).
    Se rellena con el promedio de los vecinos fluidos para que no aparezcan
    saltos artificiales (la difusión ya ignora estas celdas)."""
    pared = ~fluido & ~fuente
    suma = np.zeros_like(T)
    cuenta = np.zeros_like(T)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            m = np.roll(fluido, (dy, dx), axis=(0, 1))
            suma += np.roll(T * fluido, (dy, dx), axis=(0, 1))
            cuenta += m
    relleno = np.where(cuenta > 0, suma / np.maximum(cuenta, 1), T)
    return np.where(pared, relleno, T)


def paso_temperatura(T, rho, ux, uy, fluido, fuente, T_fuente, eos, chi):
    """Un paso de Heun (RK2) con dt = 1. La difusión es estable mientras
    8χ ≲ 1 (el factor 2 extra viene de la media armónica, ver
    _flujo_difusivo); la advección upwind, mientras |u| < 1, que el propio
    lattice Boltzmann ya exige con holgura."""
    conductor = fluido | fuente
    rho_seguro = np.where(fluido, rho, 0.1)  # valor de relleno fuera del fluido: nunca se usa, pero evita x = bρ/4 = 1
    div_u = gradiente_isotropico(ux)[0] + gradiente_isotropico(uy)[1]

    T0 = rellenar_paredes(np.where(fuente, T_fuente, T), fluido, fuente)
    k1 = lado_derecho(T0, rho_seguro, ux, uy, div_u, fluido, conductor, eos, chi)
    T1 = rellenar_paredes(T0 + k1, fluido, fuente)
    k2 = lado_derecho(T1, rho_seguro, ux, uy, div_u, fluido, conductor, eos, chi)
    return T0 + 0.5 * (k1 + k2)
