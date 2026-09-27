"""El mismo paso de tiempo que aguacero/mundo.py + aguacero/termico.py,
reescrito como bucles explícitos compilados con numba.

Por qué existe: la versión numpy es la de referencia (legible, cada línea
es una ecuación), pero hace ~40 pasadas completas por memoria por paso
(cada np.roll, cada np.where crea un arreglo nuevo). Aquí las colisiones,
la fuerza y la propagación ocurren en UNA pasada por nodo, en paralelo
por filas. Resultado medido en 160×120: ~11 ms/paso (numpy) → ~0.5-1
ms/paso (numba, 4 núcleos), que es la diferencia entre mirar una
diapositiva y jugar con agua en tiempo real.

Contrato: tests/test_numba_vs_numpy.py exige que ambos coincidan a ~1e-10
después de cientos de pasos con paredes, gravedad y fuentes de calor. Si
cambias la física, cámbiala en los dos lados.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

_EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
_EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
_W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
_OP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


@njit(cache=True, inline="always")
def _presion(rho, T, a, b, R):
    x = b * rho / 4.0
    return rho * R * T * (1.0 + x + x * x - x * x * x) / (1.0 - x) ** 3 - a * rho * rho


@njit(cache=True, inline="always")
def _dp_dT(rho, b, R):
    x = b * rho / 4.0
    return rho * R * (1.0 + x + x * x - x * x * x) / (1.0 - x) ** 3


@njit(cache=True, parallel=True)
def _gradiente(campo, gx, gy):
    ny, nx = campo.shape
    for y in prange(ny):
        for x in range(nx):
            sx = 0.0
            sy = 0.0
            for i in range(1, 9):
                v = campo[(y + _EY[i]) % ny, (x + _EX[i]) % nx]
                sx += _W[i] * v * _EX[i]
                sy += _W[i] * v * _EY[i]
            gx[y, x] = 3.0 * sx
            gy[y, x] = 3.0 * sy


@njit(cache=True, parallel=True)
def _rellenar(T, fluido, fuente, salida):
    ny, nx = T.shape
    for y in prange(ny):
        for x in range(nx):
            if fluido[y, x] or fuente[y, x]:
                salida[y, x] = T[y, x]
                continue
            suma = 0.0
            cuenta = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    yy = (y + dy) % ny
                    xx = (x + dx) % nx
                    if fluido[yy, xx]:
                        suma += T[yy, xx]
                        cuenta += 1
            salida[y, x] = suma / cuenta if cuenta > 0 else T[y, x]


@njit(cache=True, parallel=True)
def _lado_derecho(T, rho_s, ux, uy, div_u, fluido, conductor, a, b, R, cv, chi, salida):
    ny, nx = T.shape
    for y in prange(ny):
        for x in range(nx):
            if not fluido[y, x]:
                salida[y, x] = 0.0
                continue
            tc = T[y, x]
            uxc = ux[y, x]
            uyc = uy[y, x]
            adv = 0.0
            if uxc > 0.0:
                adv -= uxc * (tc - T[y, (x - 1) % nx])
            else:
                adv -= uxc * (T[y, (x + 1) % nx] - tc)
            if uyc > 0.0:
                adv -= uyc * (tc - T[(y - 1) % ny, x])
            else:
                adv -= uyc * (T[(y + 1) % ny, x] - tc)
            flujo = 0.0
            for k in range(1, 5):  # los 4 vecinos de eje son las direcciones 1-4 de D2Q9
                yy = (y + _EY[k]) % ny
                xx = (x + _EX[k]) % nx
                if conductor[y, x] and conductor[yy, xx]:
                    if fluido[yy, xx]:
                        rc = 2.0 * rho_s[y, x] * rho_s[yy, xx] / (rho_s[y, x] + rho_s[yy, xx])
                    else:
                        rc = rho_s[y, x]
                    flujo += rc * (T[yy, xx] - tc)
            dif = chi * flujo / rho_s[y, x]
            comp = -T[y, x] * _dp_dT(rho_s[y, x], b, R) / (rho_s[y, x] * cv) * div_u[y, x]
            salida[y, x] = adv + dif + comp


@njit(cache=True, parallel=True)
def _fluido(f, solido, T, tau, g, sigma_li, a, b, R, T_min, T_max, psi_pared, f_nuevo, rho, ux, uy, psi):
    _, ny, nx = f.shape
    for y in prange(ny):
        for x in range(nx):
            r = 0.0
            for i in range(9):
                r += f[i, y, x]
            rho[y, x] = r
            if solido[y, x]:
                psi[y, x] = psi_pared
            else:
                Tp = min(max(T[y, x], T_min), T_max)
                exceso = _presion(r, Tp, a, b, R) - r / 3.0
                psi[y, x] = np.sqrt(max(-6.0 * exceso, 0.0))

    for y in prange(ny):
        for x in range(nx):
            if solido[y, x]:
                ux[y, x] = 0.0
                uy[y, x] = 0.0
                for i in range(9):
                    f_nuevo[i, (y + _EY[i]) % ny, (x + _EX[i]) % nx] = f[i, y, x]
                continue
            sx = 0.0
            sy = 0.0
            for i in range(1, 9):
                v = psi[(y + _EY[i]) % ny, (x + _EX[i]) % nx]
                sx += _W[i] * v * _EX[i]
                sy += _W[i] * v * _EY[i]
            r = rho[y, x]
            Fx_coh = psi[y, x] * sx  # −G ψ Σ… con G = −1
            Fy_coh = psi[y, x] * sy
            Fx = Fx_coh
            Fy = Fy_coh + r * g
            Q = sigma_li * (Fx_coh * Fx_coh + Fy_coh * Fy_coh) / (max(psi[y, x] * psi[y, x], 1e-30) * tau)
            jx = 0.0
            jy = 0.0
            for i in range(9):
                jx += f[i, y, x] * _EX[i]
                jy += f[i, y, x] * _EY[i]
            u = (jx + 0.5 * Fx) / r
            v_ = (jy + 0.5 * Fy) / r
            ux[y, x] = u
            uy[y, x] = v_
            u2 = u * u + v_ * v_
            for i in range(9):
                eu = _EX[i] * u + _EY[i] * v_
                feq = _W[i] * r * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)
                s = (1.0 - 0.5 / tau) * _W[i] * (
                    3.0 * ((_EX[i] - u) * Fx + (_EY[i] - v_) * Fy) + 9.0 * eu * (_EX[i] * Fx + _EY[i] * Fy)
                )
                c_li = 3.0 * Q * _W[i] * (3.0 * (_EX[i] * _EX[i] + _EY[i] * _EY[i]) - 2.0)
                f_nuevo[i, (y + _EY[i]) % ny, (x + _EX[i]) % nx] = f[i, y, x] - (f[i, y, x] - feq) / tau + s + c_li

    for y in prange(ny):
        for x in range(nx):
            if solido[y, x]:
                tmp = np.empty(9)
                for i in range(9):
                    tmp[i] = f_nuevo[_OP[i], y, x]
                for i in range(9):
                    f_nuevo[i, y, x] = tmp[i]


def paso(mundo) -> None:
    p = mundo.p
    eos = p.eos
    Tc = eos.T_critica
    ny, nx = mundo.alto, mundo.ancho
    fluido = ~mundo.solido
    rho_pared = mundo.rho_vapor + p.mojabilidad * (mundo.rho_liquido - mundo.rho_vapor)
    psi_pared = float(eos.psi(np.asarray(rho_pared), mundo.T0))

    f_nuevo = np.empty_like(mundo.f)
    rho = np.empty((ny, nx))
    ux = np.empty((ny, nx))
    uy = np.empty((ny, nx))
    psi = np.empty((ny, nx))
    _fluido(
        mundo.f, mundo.solido, mundo.T, p.tau, p.gravedad, p.sigma_li, eos.a, eos.b, eos.R,
        p.T_min_reducida * Tc, p.T_max_reducida * Tc, psi_pared, f_nuevo, rho, ux, uy, psi,
    )

    mundo.f = f_nuevo
    mundo.rho, mundo.ux, mundo.uy = rho, ux, uy
    mundo.pasos += 1
    if p.isotermico:
        return

    # temperatura: Heun, idéntico a termico.paso_temperatura
    conductor = fluido | mundo.fuente
    rho_s = np.where(fluido, rho, 0.1)
    gx = np.empty_like(rho)
    gy = np.empty_like(rho)
    _gradiente(ux, gx, gy)
    div_u = gx.copy()
    _gradiente(uy, gx, gy)
    div_u += gy

    T0 = np.empty_like(rho)
    _rellenar(np.where(mundo.fuente, mundo.T_fuente, mundo.T), fluido, mundo.fuente, T0)
    k1 = np.empty_like(rho)
    _lado_derecho(T0, rho_s, ux, uy, div_u, fluido, conductor, eos.a, eos.b, eos.R, eos.cv, p.chi, k1)
    T1 = np.empty_like(rho)
    _rellenar(T0 + k1, fluido, mundo.fuente, T1)
    k2 = np.empty_like(rho)
    _lado_derecho(T1, rho_s, ux, uy, div_u, fluido, conductor, eos.a, eos.b, eos.R, eos.cv, p.chi, k2)

    mundo.T = T0 + 0.5 * (k1 + k2)
