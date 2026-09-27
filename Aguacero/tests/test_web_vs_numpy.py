"""web/nucleo.js es la tercera implementación del paso (la que corre en el
navegador). Se exporta un estado de la referencia numpy, se avanza en node
y se compara. También se compara la herramienta agregar_liquido, que en JS
reimplementa el desenfoque de scipy.ndimage.gaussian_filter."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aguacero.mundo import Mundo
from tests.test_conservacion import _escena_completa

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node no está instalado")
WEB = Path(__file__).resolve().parent.parent / "web"


def _parametros_js(m: Mundo) -> dict:
    p, eos = m.p, m.p.eos
    Tc = eos.T_critica
    return dict(
        tau=p.tau, gravedad=p.gravedad, sigma_li=p.sigma_li, chi=p.chi, mojabilidad=p.mojabilidad,
        T0=m.T0, Tc=Tc, Tmin=p.T_min_reducida * Tc, Tmax=p.T_max_reducida * Tc,
        rhoVapor=m.rho_vapor, rhoLiquido=m.rho_liquido, a=eos.a, b=eos.b, R=eos.R, cv=eos.cv,
        isotermico=p.isotermico, marco=False,
    )


def _correr_js(m: Mundo, pasos: int, tmp_path, **extra) -> dict:
    entrada = dict(
        nx=m.ancho, ny=m.alto, p=_parametros_js(m), pasos=pasos,
        f=m.f.reshape(9, -1).ravel().tolist(), T=m.T.ravel().tolist(),
        solido=m.solido.ravel().astype(int).tolist(), fuente=m.fuente.ravel().astype(int).tolist(),
        Tfuente=m.T_fuente.ravel().tolist(), **extra,
    )
    ruta_in, ruta_out = tmp_path / "in.json", tmp_path / "out.json"
    ruta_in.write_text(json.dumps(entrada))
    subprocess.run([NODE, str(WEB / "validar_node.js"), str(ruta_in), str(ruta_out)], check=True, timeout=300)
    return json.loads(ruta_out.read_text())


def test_paso_js_coincide_con_numpy(tmp_path):
    m = _escena_completa("numpy")
    m.paso(50)  # arranca desde un estado ya en movimiento
    salida = _correr_js(m, 300, tmp_path)
    m.paso(300)
    f_js = np.array(salida["f"]).reshape(m.f.shape)
    T_js = np.array(salida["T"]).reshape(m.T.shape)
    assert np.abs(f_js - m.f).max() < 1e-12
    assert np.abs(T_js - m.T).max() < 1e-12


def test_agregar_liquido_js_coincide_con_numpy(tmp_path):
    m = Mundo(60, 40, motor="numpy")
    fila, col = np.mgrid[0:40, 0:60]
    mascara = np.hypot(col - 30, fila - 20) < 8
    salida = _correr_js(m, 0, tmp_path, mascara_liquido=mascara.ravel().astype(int).tolist(), uy=0.03)
    m.agregar_liquido(mascara, uy=0.03)
    f_js = np.array(salida["f"]).reshape(m.f.shape)
    assert np.abs(f_js - m.f).max() < 1e-14
    assert salida["masaAgregada"] == pytest.approx(m.masa_agregada, rel=1e-12)
