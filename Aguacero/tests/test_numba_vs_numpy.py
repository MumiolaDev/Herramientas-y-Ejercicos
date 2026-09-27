"""aguacero/nucleo_numba.py es una reescritura del paso de mundo.py +
termico.py. Deben coincidir al redondeo en una escena con todo activo."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from tests.test_conservacion import _escena_completa  # noqa: E402


def test_mismos_resultados():
    a = _escena_completa("numpy")
    b = _escena_completa("numba")
    a.paso(400)
    b.paso(400)
    assert np.abs(a.f - b.f).max() < 1e-12
    assert np.abs(a.T - b.T).max() < 1e-12
    assert np.abs(a.ux - b.ux).max() < 1e-12
