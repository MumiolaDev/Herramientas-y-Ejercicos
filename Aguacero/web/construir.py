"""Arma la página autocontenida de la versión web: plantilla.html +
nucleo.js + los parámetros físicos calculados por la referencia Python
(densidades de coexistencia, T_c…) + las paletas de matplotlib.

    python web/construir.py salidas/aguacero.html [salidas/gota_600.npz ...]
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
        tasas=p.tasas().tolist(), gravedad=p.gravedad, sigma_li=p.sigma_li, chi=p.chi, mojabilidad=p.mojabilidad,
        T0=m.T0, Tc=Tc, Tmin=p.T_min_reducida * Tc, Tmax=p.T_max_reducida * Tc,
        rhoVapor=m.rho_vapor, rhoLiquido=m.rho_liquido, rhoCritica=eos.rho_critica,
        a=eos.a, b=eos.b, R=eos.R, cv=eos.cv, isotermico=False, marco=True,
    )
    luts = {
        nombre: (colormaps[nombre](np.linspace(0, 1, 256))[:, :3] * 255).round().astype(int).ravel().tolist()
        for nombre in ("Blues", "inferno", "viridis", "magma")
    }
    return dict(ancho=m.ancho, alto=m.alto, parametros=parametros, luts=luts)


SIGMA_07 = 6.18e-3  # tensión superficial medida a 0.7 T_c (tests/test_laplace.py)

NOTAS = {
    "gota": "La gota cae a la velocidad de caída libre que permite el número de Mach; al impactar, el vapor atrapado debajo sale expulsado y se levanta una corona con chorros laterales. Las ondas capilares recorren la piscina varias veces.",
    "grifo": "Una entrada de velocidad dentro de la tubería impone un caudal constante. El agua se acumula en la repisa, desborda por ambos bordes, cae y llena el recipiente. El |u| máximo corresponde a un punto fijo en la esquina donde la entrada toca la pared del tubo; el resto del flujo va bajo 0.1.",
    "tetera": "Placa a 1.1 T_c bajo una piscina a 0.7 T_c. El domo de vapor crece por calentamiento hasta que la inestabilidad de Rayleigh-Taylor lo hace atravesar el líquido; el ciclo se repite. El calentamiento es difusivo, por eso esta corrida usa menos resolución.",
}
TITULOS = {"gota": "Gota", "grifo": "Grifo", "tetera": "Tetera"}


def datos_render(ruta_npz: str, fps: int = 30) -> dict:
    d = np.load(ruta_npz)
    escena = str(d["escena"])
    par = dict(zip([str(c) for c in d["columnas_parametros"]], d["parametros"]))
    serie = dict(zip([str(c) for c in d["columnas_serie"]], d["serie"].T))
    n, alto, ancho = d["agua"].shape
    lc = np.sqrt(SIGMA_07 / ((par["rho_liquido"] - par["rho_vapor"]) * par["gravedad"]))
    oh = par["rho_liquido"] * par["viscosidad"] / np.sqrt(par["rho_liquido"] * SIGMA_07 * lc)
    ficha = [
        ["Resolución", f"{ancho}×{alto}"],
        ["Pasos", f"{int(d['pasos'][-1]):,}".replace(",", ".")],
        ["Cómputo", f"{float(d['segundos']) / 60:.1f} min"],
        ["Viscosidad ν", f"{par['viscosidad']:.4f}"],
        ["Gravedad g", f"{par['gravedad']:.2e}"],
        ["Longitud capilar", f"{lc:.0f} px"],
        ["Ohnesorge Oh(l_c)", f"{oh:.3f}"],
        ["|u| máximo", f"{serie['umax'].max():.3f}"],
    ]
    if escena == "grifo":
        ficha.append(["Masa agregada", f"{serie['masa_agregada'][-1]:.0f}"])
    nombre = Path(ruta_npz).stem
    return dict(
        id=nombre, titulo=f"{TITULOS.get(escena, escena)} {ancho}×{alto}", video=f"videos/{nombre}", fps=fps,
        pasos=d["pasos"].tolist(), cinetica=serie["cinetica"].round(6).tolist(),
        interna=serie["interna"].round(4).tolist(), umax=serie["umax"].round(5).tolist(),
        ficha=ficha, nota=NOTAS.get(escena, ""),
    )


def main(destino: str, renders: list[str] | None = None) -> None:
    html = (AQUI / "plantilla.html").read_text(encoding="utf-8")
    html = html.replace("/*__NUCLEO__*/", (AQUI / "nucleo.js").read_text(encoding="utf-8"))
    html = html.replace("/*__DATOS__*/", json.dumps(datos(), separators=(",", ":")))
    lista = [datos_render(r) for r in (renders or [])]
    html = html.replace("/*__RENDERS__*/", json.dumps(lista, separators=(",", ":")))
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    Path(destino).write_text(html, encoding="utf-8")
    print(f"guardado: {destino} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    # python web/construir.py salida.html [render1.npz render2.npz ...]
    # Los videos se esperan publicados junto a la página como videos/<nombre>.mp4.
    main(sys.argv[1] if len(sys.argv) > 1 else "salidas/aguacero.html", sys.argv[2:])
