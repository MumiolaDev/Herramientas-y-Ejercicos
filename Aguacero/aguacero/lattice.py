"""La red D2Q9: nueve velocidades discretas por nodo (reposo, 4 ejes, 4
diagonales). Todo lo que sigue — momentos, equilibrio, fuerza — son sumas
sobre estos nueve vectores, que es exactamente lo que hace de lattice
Boltzmann un autómata celular: cada pixel sólo necesita a sus 8 vecinos.

Convención de ejes: arreglos indexados como [fila, columna] con la fila
creciendo HACIA ABAJO (la convención de pantalla, para que pygame no
tenga que voltear nada). Por eso la gravedad apunta a +fila y las
componentes de velocidad son (ux → columnas, uy → filas).
"""

from __future__ import annotations

import numpy as np

# (ex, ey) con ey positivo = hacia abajo en pantalla
E = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPUESTO = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
CS2 = 1.0 / 3.0  # velocidad del sonido al cuadrado, en unidades de red

EX = E[:, 0].astype(float)[:, None, None]
EY = E[:, 1].astype(float)[:, None, None]
W3 = W[:, None, None]


# Matriz de momentos de Lallemand & Luo (2000) para este orden de velocidades.
# Filas: ρ, e (energía), ε (energía²), jx, qx (flujo de energía), jy, qy, pxx, pxy.
# Las filas son ortogonales, así que M⁻¹ = Mᵀ D⁻¹ con D = diag(Σ fila²).
M = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
        [4, -2, -2, -2, -2, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, -2, 0, 2, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 0, -2, 0, 2, 1, 1, -1, -1],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
    ],
    dtype=float,
)
M_INV = M.T / (M**2).sum(axis=1)


def equilibrio(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Maxwell-Boltzmann truncada a segundo orden en u (lo mínimo para
    recuperar Navier-Stokes vía Chapman-Enskog)."""
    eu = EX * ux + EY * uy
    u2 = ux * ux + uy * uy
    return W3 * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)


def desplazar(campo: np.ndarray, i: int) -> np.ndarray:
    """campo(x + e_i): el valor del vecino en la dirección i."""
    return np.roll(campo, (-E[i, 1], -E[i, 0]), axis=(0, 1))


def propagar(f: np.ndarray) -> np.ndarray:
    """Streaming: cada población viaja un nodo en su dirección. Es una
    permutación de datos — no hay aritmética, así que no puede crear ni
    destruir masa (ver tests/test_conservacion.py)."""
    salida = np.empty_like(f)
    for i in range(9):
        salida[i] = np.roll(f[i], (E[i, 1], E[i, 0]), axis=(0, 1))
    return salida


def gradiente_isotropico(campo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """∇φ ≈ (1/c_s²) Σ w_i φ(x+e_i) e_i: diferencia centrada que usa los
    8 vecinos con los pesos de la red. Es isótropo a segundo orden — a
    diferencia de la diferencia centrada de 4 vecinos, no privilegia los
    ejes, lo que importa porque la tensión superficial (que sale de este
    mismo tipo de suma) también debe ser isótropa o las gotas se vuelven
    cuadradas."""
    gx = np.zeros_like(campo)
    gy = np.zeros_like(campo)
    for i in range(1, 9):
        vecino = desplazar(campo, i)
        gx += W[i] * vecino * E[i, 0]
        gy += W[i] * vecino * E[i, 1]
    return gx / CS2, gy / CS2
