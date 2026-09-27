"""El mismo paso de tiempo que aguacero/mundo.py + aguacero/termico.py,
reescrito como bucles explícitos compilados con numba.

Por qué existe: la versión numpy es la de referencia (legible, cada línea
es una ecuación), pero hace decenas de pasadas completas por memoria por
paso (cada np.roll, cada np.where crea un arreglo nuevo). Aquí la fuerza,
la colisión MRT y la propagación ocurren en UNA pasada por nodo, y la
ecuación de energía en cuatro kernels sin arreglos temporales, todo en
paralelo por filas y sobre búferes que se reutilizan entre pasos.

Contrato: tests/test_numba_vs_numpy.py exige que ambos coincidan a 1e-12
después de cientos de pasos con paredes, gravedad y fuentes de calor. Si
cambias la física, cámbiala en los dos lados.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from aguacero.lattice import M as _M_FLOAT
from aguacero.lattice import M_INV as _M_INV

_EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
_EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
_W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
_OP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
_M = np.ascontiguousarray(_M_FLOAT)
_MI = np.ascontiguousarray(_M_INV)


@njit(cache=True, inline="always")
def _presion(rho, T, a, b, R):
    x = b * rho / 4.0
    return rho * R * T * (1.0 + x + x * x - x * x * x) / (1.0 - x) ** 3 - a * rho * rho


@njit(cache=True, inline="always")
def _dp_dT(rho, b, R):
    x = b * rho / 4.0
    return rho * R * (1.0 + x + x * x - x * x * x) / (1.0 - x) ** 3


# --------------------------------------------------------------------- fluido


@njit(cache=True, parallel=True)
def _fluido(f, solido, T, S, g, sigma_li, a, b, R, T_min, T_max, psi_pared, f_nuevo, rho, ux, uy, psi):
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
        m = np.empty(9)
        mc = np.empty(9)
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
            Fxc = psi[y, x] * sx  # −G ψ Σ… con G = −1
            Fyc = psi[y, x] * sy
            Fx = Fxc
            Fy = Fyc + r * g
            X = sigma_li * (Fxc * Fxc + Fyc * Fyc) / max(psi[y, x] * psi[y, x], 1e-30)

            for k in range(9):
                acc = 0.0
                for i in range(9):
                    acc += _M[k, i] * f[i, y, x]
                m[k] = acc
            u = (m[3] + 0.5 * Fx) / r
            v_ = (m[5] + 0.5 * Fy) / r
            ux[y, x] = u
            uy[y, x] = v_
            u2 = u * u + v_ * v_
            uF = u * Fx + v_ * Fy
            # m* = m − S(m − m_eq) + (1 − S/2) G + L
            mc[0] = m[0] - S[0] * (m[0] - r)
            mc[1] = m[1] - S[1] * (m[1] - r * (-2.0 + 3.0 * u2)) + (1.0 - 0.5 * S[1]) * 6.0 * uF + 12.0 * S[1] * X
            mc[2] = m[2] - S[2] * (m[2] - r * (1.0 - 3.0 * u2)) - (1.0 - 0.5 * S[2]) * 6.0 * uF - 12.0 * S[2] * X
            mc[3] = m[3] - S[3] * (m[3] - r * u) + (1.0 - 0.5 * S[3]) * Fx
            mc[4] = m[4] - S[4] * (m[4] + r * u) - (1.0 - 0.5 * S[4]) * Fx
            mc[5] = m[5] - S[5] * (m[5] - r * v_) + (1.0 - 0.5 * S[5]) * Fy
            mc[6] = m[6] - S[6] * (m[6] + r * v_) - (1.0 - 0.5 * S[6]) * Fy
            mc[7] = m[7] - S[7] * (m[7] - r * (u * u - v_ * v_)) + (1.0 - 0.5 * S[7]) * 2.0 * (u * Fx - v_ * Fy)
            mc[8] = m[8] - S[8] * (m[8] - r * u * v_) + (1.0 - 0.5 * S[8]) * (u * Fy + v_ * Fx)
            for i in range(9):
                acc = 0.0
                for k in range(9):
                    acc += _MI[i, k] * mc[k]
                f_nuevo[i, (y + _EY[i]) % ny, (x + _EX[i]) % nx] = acc

    for y in prange(ny):
        tmp = np.empty(9)
        for x in range(nx):
            if solido[y, x]:
                for i in range(9):
                    tmp[i] = f_nuevo[_OP[i], y, x]
                for i in range(9):
                    f_nuevo[i, y, x] = tmp[i]


# ---------------------------------------------------------------- temperatura


@njit(cache=True, parallel=True)
def _divergencia(ux, uy, salida):
    """∂x ux + ∂y uy con el gradiente isótropo de 8 vecinos. Se suman por
    separado las dos derivadas (como numpy) para coincidir al redondeo."""
    ny, nx = ux.shape
    for y in prange(ny):
        for x in range(nx):
            gx = 0.0
            gy = 0.0
            for i in range(1, 9):
                yy = (y + _EY[i]) % ny
                xx = (x + _EX[i]) % nx
                gx += _W[i] * ux[yy, xx] * _EX[i]
                gy += _W[i] * uy[yy, xx] * _EY[i]
            salida[y, x] = 3.0 * gx + 3.0 * gy


@njit(cache=True, inline="always")
def _valor(A, B, c, fuente, T_fuente, usar_fuente, y, x):
    # A + c·B, o la temperatura impuesta si la celda es fuente y así se pide
    if usar_fuente and fuente[y, x]:
        return T_fuente[y, x]
    return A[y, x] + c * B[y, x]


@njit(cache=True, parallel=True)
def _rellenar(A, B, c, fluido, fuente, T_fuente, usar_fuente, salida):
    """salida = A + c·B en fluido y fuentes; en paredes adiabáticas, el
    promedio de los vecinos fluidos (termico.rellenar_paredes)."""
    ny, nx = A.shape
    for y in prange(ny):
        for x in range(nx):
            if fluido[y, x] or fuente[y, x]:
                salida[y, x] = _valor(A, B, c, fuente, T_fuente, usar_fuente, y, x)
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
                        suma += _valor(A, B, c, fuente, T_fuente, usar_fuente, yy, xx)
                        cuenta += 1
            salida[y, x] = suma / cuenta if cuenta > 0 else _valor(A, B, c, fuente, T_fuente, usar_fuente, y, x)


@njit(cache=True, parallel=True)
def _lado_derecho(T, rho, ux, uy, div_u, fluido, fuente, b, R, cv, chi, salida):
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
            r = rho[y, x]
            flujo = 0.0
            for k in range(1, 5):  # los 4 vecinos de eje son las direcciones 1-4 de D2Q9
                yy = (y + _EY[k]) % ny
                xx = (x + _EX[k]) % nx
                if fluido[yy, xx]:
                    rc = 2.0 * r * rho[yy, xx] / (r + rho[yy, xx])
                elif fuente[yy, xx]:
                    rc = r
                else:
                    continue
                flujo += rc * (T[yy, xx] - tc)
            dif = chi * flujo / r
            comp = -tc * _dp_dT(r, b, R) / (r * cv) * div_u[y, x]
            salida[y, x] = adv + dif + comp


@njit(cache=True, parallel=True)
def _heun_final(T0, k1, k2, salida):
    ny, nx = T0.shape
    for y in prange(ny):
        for x in range(nx):
            salida[y, x] = T0[y, x] + 0.5 * (k1[y, x] + k2[y, x])


# ------------------------------------------------------------------- paso


def _bufer(mundo, nombre, forma):
    buf = mundo.__dict__.setdefault("_bufer_numba", {})
    arr = buf.get(nombre)
    if arr is None or arr.shape != forma:
        arr = np.empty(forma)
        buf[nombre] = arr
    return arr


def paso(mundo) -> None:
    p = mundo.p
    eos = p.eos
    Tc = eos.T_critica
    forma = (mundo.alto, mundo.ancho)
    rho_pared = mundo.rho_vapor + p.mojabilidad * (mundo.rho_liquido - mundo.rho_vapor)
    psi_pared = float(eos.psi(np.asarray(rho_pared), mundo.T0))

    # doble búfer para f: el viejo se reutiliza como destino del siguiente paso
    f_nuevo = _bufer(mundo, "f_nuevo", mundo.f.shape)
    rho = np.empty(forma)
    ux = np.empty(forma)
    uy = np.empty(forma)
    psi = _bufer(mundo, "psi", forma)
    _fluido(
        mundo.f, mundo.solido, mundo.T, p.tasas(), p.gravedad, p.sigma_li, eos.a, eos.b, eos.R,
        p.T_min_reducida * Tc, p.T_max_reducida * Tc, psi_pared, f_nuevo, rho, ux, uy, psi,
    )
    mundo.__dict__["_bufer_numba"]["f_nuevo"] = mundo.f
    mundo.f = f_nuevo
    mundo.rho, mundo.ux, mundo.uy = rho, ux, uy
    mundo.pasos += 1
    if p.isotermico:
        return

    # temperatura: Heun, idéntico a termico.paso_temperatura
    fluido = ~mundo.solido
    div_u = _bufer(mundo, "div_u", forma)
    T0 = _bufer(mundo, "T0", forma)
    T1 = _bufer(mundo, "T1", forma)
    k1 = _bufer(mundo, "k1", forma)
    k2 = _bufer(mundo, "k2", forma)
    _divergencia(ux, uy, div_u)
    _rellenar(mundo.T, mundo.T, 0.0, fluido, mundo.fuente, mundo.T_fuente, True, T0)
    _lado_derecho(T0, rho, ux, uy, div_u, fluido, mundo.fuente, eos.b, eos.R, eos.cv, p.chi, k1)
    _rellenar(T0, k1, 1.0, fluido, mundo.fuente, mundo.T_fuente, False, T1)
    _lado_derecho(T1, rho, ux, uy, div_u, fluido, mundo.fuente, eos.b, eos.R, eos.cv, p.chi, k2)
    T_nueva = np.empty(forma)
    _heun_final(T0, k1, k2, T_nueva)
    mundo.T = T_nueva
