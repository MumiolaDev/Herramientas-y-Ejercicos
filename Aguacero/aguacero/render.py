"""Render offline: correr una escena grande sin ventana, guardar los
cuadros y (opcional) exportar video.

La simulación en tiempo real obliga a cajas chicas; aquí la resolución es
la que el problema pide y el tiempo de cómputo se paga una vez. Lo que se
guarda es física, no imágenes: fracción de líquido y T/T_c cuantizadas a
8 bits por pixel, más la serie temporal de masa y energías, así que el
mismo archivo sirve para video, para una página web o para análisis.

    python -m aguacero.render grifo --ancho 600 --alto 450 --pasos 30000 --cada 60 \\
        --salida salidas/grifo_600.npz --video salidas/grifo_600.mp4

Costo medido (numba, 4 núcleos): ~75 ns por nodo y paso con temperatura,
o sea 600×450 ≈ 20 ms/paso ≈ 10 min por cada 30 000 pasos.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from aguacero.escenas import ESCENAS
from aguacero.mundo import Parametros

T_MIN, T_MAX = 0.55, 1.2  # rango de T/T_c que se cuantiza en 0-255


def parametros_escalados(alto: int, tau: float | None = None) -> Parametros:
    """Parámetros para una caja k = alto/150 veces la de referencia.

    La gravedad se escala como g/k. Razón: la velocidad de caída libre a
    través de la caja es √(2gH) y H crece ∝ k; con g fijo, a 600×450 una
    gota llega a |u| ≈ 0.15 y el vapor expulsado bajo ella a 0.44 (Mach
    0.76, medido), donde lattice Boltzmann deja de ser preciso. Con g/k la
    velocidad máxima no cambia con la resolución, y aun así el número de
    Bond de la caja (Δρ g H²/σ) crece ∝ k: la caja grande sigue siendo
    físicamente más "macroscópica". La longitud capilar pasa a 19√k
    pixeles."""
    k = alto / 150.0
    base = Parametros() if tau is None else Parametros(tau=tau)
    base.gravedad = base.gravedad / k
    return base


def renderizar(
    escena: str,
    ancho: int,
    alto: int,
    pasos: int,
    cada: int,
    parametros: Parametros | None = None,
    motor: str = "auto",
    progreso: bool = True,
) -> dict:
    mundo, cada_paso = ESCENAS[escena](ancho=ancho, alto=alto, motor=motor, parametros=parametros)
    Tc = mundo.p.eos.T_critica
    agua, temp, t, serie = [], [], [], []
    t0 = time.perf_counter()

    def guardar():
        agua.append(np.round(mundo.fraccion_liquido() * 255).astype(np.uint8))
        temp.append(np.round(np.clip((mundo.T / Tc - T_MIN) / (T_MAX - T_MIN), 0, 1) * 255).astype(np.uint8))
        e = mundo.energias()
        t.append(mundo.pasos)
        serie.append([mundo.masa_total(), mundo.masa_agregada, e["cinetica"], e["potencial"], e["interna"],
                      float(np.hypot(mundo.ux, mundo.uy).max())])

    guardar()
    while mundo.pasos < pasos:
        if cada_paso is not None:
            cada_paso(mundo)
        mundo.paso(1)
        if mundo.pasos % cada == 0:
            if not np.isfinite(mundo.f).all():
                raise FloatingPointError(f"la simulación divergió antes del paso {mundo.pasos}")
            guardar()
            if progreso and len(t) % 20 == 0:
                dt = time.perf_counter() - t0
                eta = dt / mundo.pasos * (pasos - mundo.pasos)
                print(f"  paso {mundo.pasos}/{pasos}  {1e3 * dt / mundo.pasos:.1f} ms/paso  faltan ~{eta / 60:.1f} min", flush=True)

    p = mundo.p
    return dict(
        escena=escena,
        agua=np.stack(agua),
        temperatura=np.stack(temp),
        solido=mundo.solido.copy(),
        fuente=mundo.fuente.copy(),
        fuente_caliente=(mundo.fuente & (mundo.T_fuente > mundo.T0)),
        pasos=np.array(t),
        serie=np.array(serie),
        columnas_serie=np.array(["masa", "masa_agregada", "cinetica", "potencial", "interna", "umax"]),
        T_rango=np.array([T_MIN, T_MAX]),
        parametros=np.array(
            [p.tau, p.viscosidad, p.gravedad, p.chi, p.T_reducida, p.sigma_li, mundo.rho_liquido, mundo.rho_vapor]
        ),
        columnas_parametros=np.array(["tau", "viscosidad", "gravedad", "chi", "T_reducida", "sigma_li", "rho_liquido", "rho_vapor"]),
        segundos=time.perf_counter() - t0,
    )


def _lut(nombre: str) -> np.ndarray:
    from matplotlib import colormaps

    return (colormaps[nombre](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def colorear(datos: dict, campo: str) -> np.ndarray:
    """Cuadros RGB (n, alto, ancho, 3) de 'agua' o 'temperatura'."""
    if campo == "agua":
        idx = (20 + (datos["agua"].astype(np.uint16) * 235) // 255).astype(np.uint8)
        rgb = _lut("Blues")[idx]
    else:
        rgb = _lut("inferno")[datos["temperatura"]]
    rgb[:, datos["solido"]] = (70, 72, 80)
    rgb[:, datos["fuente_caliente"]] = (224, 88, 46)
    rgb[:, datos["fuente"] & ~datos["fuente_caliente"]] = (60, 170, 230)
    return rgb


def exportar_video(datos: dict, ruta: str, fps: int = 30) -> None:
    """Video con el agua a la izquierda y la temperatura a la derecha
    (separadas por una franja de 4 pixeles). Escribe `ruta`
    (.mp4, H.264: Safari/iOS) y al lado un .webm (VP9: Chrome, Firefox,
    Android); ningún códec solo lo reproduce en todos los navegadores."""
    import imageio.v2 as imageio

    agua, temp = colorear(datos, "agua"), colorear(datos, "temperatura")
    franja = np.full(agua.shape[:2] + (4, 3), 20, dtype=np.uint8)
    cuadros = np.concatenate([agua, franja, temp], axis=2)
    # H.264 exige dimensiones pares
    alto, ancho = cuadros.shape[1] - cuadros.shape[1] % 2, cuadros.shape[2] - cuadros.shape[2] % 2
    cuadros = cuadros[:, :alto, :ancho]
    base = ruta[: -len(".mp4")] if ruta.endswith(".mp4") else ruta
    formatos = [
        (base + ".mp4", dict(codec="libx264", quality=8, ffmpeg_params=["-movflags", "+faststart"])),
        (base + ".webm", dict(codec="libvpx-vp9", ffmpeg_params=["-b:v", "0", "-crf", "34", "-row-mt", "1"])),
    ]
    for destino, opciones in formatos:
        with imageio.get_writer(destino, fps=fps, pixelformat="yuv420p", macro_block_size=1, **opciones) as w:
            for c in cuadros:
                w.append_data(c)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m aguacero.render", description=__doc__.split("\n\n")[0])
    ap.add_argument("escena", choices=sorted(ESCENAS))
    ap.add_argument("--ancho", type=int, default=600)
    ap.add_argument("--alto", type=int, default=450)
    ap.add_argument("--pasos", type=int, default=30000)
    ap.add_argument("--cada", type=int, default=60, help="guardar un cuadro cada N pasos")
    ap.add_argument("--tau", type=float, default=None, help="τ de los esfuerzos (viscosidad); por defecto el de Parametros")
    ap.add_argument("--salida", default=None, help="archivo .npz (por defecto salidas/<escena>_<ancho>.npz)")
    ap.add_argument("--video", default=None, help="exportar además un .mp4")
    args = ap.parse_args(argv)

    parametros = parametros_escalados(args.alto, args.tau)
    print(f"{args.escena} {args.ancho}×{args.alto}, {args.pasos} pasos, ν = {parametros.viscosidad:.4f}, g = {parametros.gravedad:.2e}")
    datos = renderizar(args.escena, args.ancho, args.alto, args.pasos, args.cada, parametros)
    salida = args.salida or f"salidas/{args.escena}_{args.ancho}.npz"
    import os

    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    np.savez_compressed(salida, **datos)
    print(f"guardado: {salida}  ({datos['agua'].shape[0]} cuadros, {datos['segundos'] / 60:.1f} min)")
    if args.video:
        exportar_video(datos, args.video)
        print(f"video: {args.video} (+ .webm)")


if __name__ == "__main__":
    main()
