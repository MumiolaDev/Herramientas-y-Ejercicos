"""Arma la página autocontenida de la versión web: plantilla.html +
nucleo.js + los parámetros físicos calculados por la referencia Python
(densidades de coexistencia, T_c…) + las paletas de matplotlib.

    python web/construir.py salidas/aguacero.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps

from aguacero.mundo import Mundo

AQUI = Path(__file__).resolve().parent


def datos() -> dict:
    m = Mundo(200, 150, motor="numpy")
    p, eos = m.p, m.p.eos
    Tc = eos.T_critica
    parametros = dict(
        tau=p.tau, gravedad=p.gravedad, sigma_li=p.sigma_li, chi=p.chi, mojabilidad=p.mojabilidad,
        T0=m.T0, Tc=Tc, Tmin=p.T_min_reducida * Tc, Tmax=p.T_max_reducida * Tc,
        rhoVapor=m.rho_vapor, rhoLiquido=m.rho_liquido, rhoCritica=eos.rho_critica,
        a=eos.a, b=eos.b, R=eos.R, cv=eos.cv, isotermico=False, marco=True,
    )
    luts = {
        nombre: (colormaps[nombre](np.linspace(0, 1, 256))[:, :3] * 255).round().astype(int).ravel().tolist()
        for nombre in ("Blues", "inferno", "viridis", "magma")
    }
    return dict(ancho=m.ancho, alto=m.alto, parametros=parametros, luts=luts)


def main(destino: str) -> None:
    html = (AQUI / "plantilla.html").read_text(encoding="utf-8")
    html = html.replace("/*__NUCLEO__*/", (AQUI / "nucleo.js").read_text(encoding="utf-8"))
    html = html.replace("/*__DATOS__*/", json.dumps(datos(), separators=(",", ":")))
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    Path(destino).write_text(html, encoding="utf-8")
    print(f"guardado: {destino} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "salidas/aguacero.html")
