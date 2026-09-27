// Aguacero — el mismo paso de tiempo que aguacero/nucleo_numba.py, en
// JavaScript, para correr en el navegador (web/plantilla.html) o en node
// (tests/test_web_vs_numpy.py lo compara contra la referencia numpy).
//
// Si cambias la física, cámbiala en los tres motores: numpy, numba y este.

(function (raiz) {
  "use strict";

  const EX = [0, 1, 0, -1, 0, 1, -1, -1, 1];
  const EY = [0, 0, 1, 0, -1, 1, 1, -1, -1];
  const W = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
  const OP = [0, 3, 4, 1, 2, 7, 8, 5, 6];
  // matriz de momentos de Lallemand & Luo (idéntica a aguacero/lattice.py) y su inversa Mᵀ D⁻¹
  const M = [
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    -4, -1, -1, -1, -1, 2, 2, 2, 2,
    4, -2, -2, -2, -2, 1, 1, 1, 1,
    0, 1, 0, -1, 0, 1, -1, -1, 1,
    0, -2, 0, 2, 0, 1, -1, -1, 1,
    0, 0, 1, 0, -1, 1, 1, -1, -1,
    0, 0, -2, 0, 2, 1, 1, -1, -1,
    0, 1, -1, 1, -1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, -1, 1, -1,
  ];
  const MI = (function () {
    const D = [9, 36, 36, 6, 12, 6, 12, 4, 4], r = new Float64Array(81);
    for (let i = 0; i < 9; i++) for (let a = 0; a < 9; a++) r[i * 9 + a] = M[a * 9 + i] / D[a];
    return r;
  })();

  function presion(rho, T, a, b, R) {
    const x = (b * rho) / 4.0;
    const d = 1.0 - x;
    return (rho * R * T * (1.0 + x + x * x - x * x * x)) / (d * d * d) - a * rho * rho;
  }

  function dpdT(rho, b, R) {
    const x = (b * rho) / 4.0;
    const d = 1.0 - x;
    return (rho * R * (1.0 + x + x * x - x * x * x)) / (d * d * d);
  }

  function psiDe(rho, T, a, b, R) {
    const exceso = presion(rho, T, a, b, R) - rho / 3.0;
    return Math.sqrt(Math.max(-6.0 * exceso, 0.0));
  }

  function crearMundo(ancho, alto, p) {
    // p: {tasas (9 tasas de relajación MRT), gravedad, sigma_li, chi, mojabilidad, T0, Tmin, Tmax,
    //     rhoVapor, rhoLiquido, a, b, R, cv, isotermico, marco}
    const nx = ancho, ny = alto, N = nx * ny;
    const m = {
      nx, ny, N, p,
      f: new Float64Array(9 * N), fn: new Float64Array(9 * N),
      rho: new Float64Array(N), ux: new Float64Array(N), uy: new Float64Array(N),
      psi: new Float64Array(N), T: new Float64Array(N),
      solido: new Uint8Array(N), fuente: new Uint8Array(N), Tfuente: new Float64Array(N),
      pasos: 0, masaAgregada: 0,
      // temporales de la ecuación de energía
      _gx: new Float64Array(N), _gy: new Float64Array(N), _div: new Float64Array(N),
      _T0: new Float64Array(N), _T1: new Float64Array(N), _k1: new Float64Array(N),
      _k2: new Float64Array(N), _tmp: new Float64Array(N), _rs: new Float64Array(N),
      // vec[i*N + k] = índice del vecino de k en la dirección i (periódico).
      // Precalculado: los % dentro de los bucles calientes costaban ~la mitad del paso.
      vec: new Int32Array(9 * N),
    };
    for (let i = 0; i < 9; i++)
      for (let y = 0; y < ny; y++)
        for (let x = 0; x < nx; x++)
          m.vec[i * N + y * nx + x] = ((y + EY[i] + ny) % ny) * nx + ((x + EX[i] + nx) % nx);
    m.T.fill(p.T0);
    m.Tfuente.fill(p.T0);
    if (p.marco !== false) {
      for (let x = 0; x < nx; x++) { m.solido[x] = 1; m.solido[(ny - 1) * nx + x] = 1; }
      for (let y = 0; y < ny; y++) { m.solido[y * nx] = 1; m.solido[y * nx + nx - 1] = 1; }
    }
    for (let k = 0; k < N; k++) ponerEquilibrio(m, k, p.rhoVapor, 0, 0);
    for (let k = 0; k < N; k++) m.rho[k] = p.rhoVapor;
    return m;
  }

  function ponerEquilibrio(m, k, r, u, v) {
    const u2 = u * u + v * v;
    for (let i = 0; i < 9; i++) {
      const eu = EX[i] * u + EY[i] * v;
      m.f[i * m.N + k] = W[i] * r * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
    }
  }

  function psiPared(m) {
    const p = m.p;
    const rp = p.rhoVapor + p.mojabilidad * (p.rhoLiquido - p.rhoVapor);
    return psiDe(rp, p.T0, p.a, p.b, p.R);
  }

  function pasoFluido(m) {
    const { nx, ny, N, f, fn, rho, ux, uy, psi, T, solido, p, vec } = m;
    const S = p.tasas, g = p.gravedad, sig = p.sigma_li;
    const m9 = new Float64Array(9), mc = new Float64Array(9);
    const pw = psiPared(m);
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const k = y * nx + x;
        let r = 0.0;
        for (let i = 0; i < 9; i++) r += f[i * N + k];
        rho[k] = r;
        if (solido[k]) psi[k] = pw;
        else {
          const Tp = Math.min(Math.max(T[k], p.Tmin), p.Tmax);
          const exceso = presion(r, Tp, p.a, p.b, p.R) - r / 3.0;
          psi[k] = Math.sqrt(Math.max(-6.0 * exceso, 0.0));
        }
      }
    }
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const k = y * nx + x;
        if (solido[k]) {
          ux[k] = 0.0; uy[k] = 0.0;
          for (let i = 0; i < 9; i++) fn[i * N + vec[i * N + k]] = f[i * N + k];
          continue;
        }
        let sx = 0.0, sy = 0.0;
        for (let i = 1; i < 9; i++) {
          const v = psi[vec[i * N + k]];
          sx += W[i] * v * EX[i];
          sy += W[i] * v * EY[i];
        }
        const r = rho[k];
        const Fxc = psi[k] * sx, Fyc = psi[k] * sy;
        const Fx = Fxc, Fy = Fyc + r * g;
        const X = (sig * (Fxc * Fxc + Fyc * Fyc)) / Math.max(psi[k] * psi[k], 1e-30);
        for (let a = 0; a < 9; a++) {
          let acc = 0.0;
          for (let i = 0; i < 9; i++) acc += M[a * 9 + i] * f[i * N + k];
          m9[a] = acc;
        }
        const u = (m9[3] + 0.5 * Fx) / r, v = (m9[5] + 0.5 * Fy) / r;
        ux[k] = u; uy[k] = v;
        const u2 = u * u + v * v, uF = u * Fx + v * Fy;
        // m* = m − S(m − m_eq) + (1 − S/2) G + L   (MRT + Guo + Li; ver aguacero/mundo.py)
        mc[0] = m9[0] - S[0] * (m9[0] - r);
        mc[1] = m9[1] - S[1] * (m9[1] - r * (-2.0 + 3.0 * u2)) + (1.0 - 0.5 * S[1]) * 6.0 * uF + 12.0 * S[1] * X;
        mc[2] = m9[2] - S[2] * (m9[2] - r * (1.0 - 3.0 * u2)) - (1.0 - 0.5 * S[2]) * 6.0 * uF - 12.0 * S[2] * X;
        mc[3] = m9[3] - S[3] * (m9[3] - r * u) + (1.0 - 0.5 * S[3]) * Fx;
        mc[4] = m9[4] - S[4] * (m9[4] + r * u) - (1.0 - 0.5 * S[4]) * Fx;
        mc[5] = m9[5] - S[5] * (m9[5] - r * v) + (1.0 - 0.5 * S[5]) * Fy;
        mc[6] = m9[6] - S[6] * (m9[6] + r * v) - (1.0 - 0.5 * S[6]) * Fy;
        mc[7] = m9[7] - S[7] * (m9[7] - r * (u * u - v * v)) + (1.0 - 0.5 * S[7]) * 2.0 * (u * Fx - v * Fy);
        mc[8] = m9[8] - S[8] * (m9[8] - r * u * v) + (1.0 - 0.5 * S[8]) * (u * Fy + v * Fx);
        for (let i = 0; i < 9; i++) {
          let acc = 0.0;
          for (let a = 0; a < 9; a++) acc += MI[i * 9 + a] * mc[a];
          fn[i * N + vec[i * N + k]] = acc;
        }
      }
    }
    const tmp = new Float64Array(9);
    for (let k = 0; k < N; k++) {
      if (!solido[k]) continue;
      for (let i = 0; i < 9; i++) tmp[i] = fn[OP[i] * N + k];
      for (let i = 0; i < 9; i++) fn[i * N + k] = tmp[i];
    }
    m.f = fn; m.fn = f;
  }

  function gradiente(m, campo, gx, gy) {
    const { N, vec } = m;
    for (let k = 0; k < N; k++) {
      let sx = 0.0, sy = 0.0;
      for (let i = 1; i < 9; i++) {
        const v = campo[vec[i * N + k]];
        sx += W[i] * v * EX[i];
        sy += W[i] * v * EY[i];
      }
      gx[k] = 3.0 * sx;
      gy[k] = 3.0 * sy;
    }
  }

  // las 8 direcciones en el orden (dy, dx) = (−1,−1), (−1,0), … (1,1) que usan
  // numpy y numba al rellenar paredes (el orden de la suma importa al redondeo)
  const ORDEN_VECINOS = [7, 4, 8, 3, 1, 6, 2, 5];

  function rellenar(m, T, salida) {
    const { N, solido, fuente, vec } = m;
    for (let k = 0; k < N; k++) {
      if (!solido[k] || fuente[k]) { salida[k] = T[k]; continue; }
      let suma = 0.0, cuenta = 0;
      for (let j = 0; j < 8; j++) {
        const kk = vec[ORDEN_VECINOS[j] * N + k];
        if (!solido[kk]) { suma += T[kk]; cuenta++; }
      }
      salida[k] = cuenta > 0 ? suma / cuenta : T[k];
    }
  }

  function ladoDerecho(m, T, salida) {
    const { N, solido, fuente, ux, uy, p, vec } = m;
    const rs = m._rs, div = m._div;
    for (let k = 0; k < N; k++) {
        if (solido[k]) { salida[k] = 0.0; continue; }
        const tc = T[k], uxc = ux[k], uyc = uy[k];
        let adv = 0.0;
        // direcciones D2Q9: 1 = +x, 2 = +y (abajo), 3 = −x, 4 = −y
        if (uxc > 0.0) adv -= uxc * (tc - T[vec[3 * N + k]]);
        else adv -= uxc * (T[vec[N + k]] - tc);
        if (uyc > 0.0) adv -= uyc * (tc - T[vec[4 * N + k]]);
        else adv -= uyc * (T[vec[2 * N + k]] - tc);
        let flujo = 0.0;
        for (let i = 1; i < 5; i++) {
          const kk = vec[i * N + k];
          const condK = !solido[k] || fuente[k];
          const condV = !solido[kk] || fuente[kk];
          if (condK && condV) {
            const rc = !solido[kk] ? (2.0 * rs[k] * rs[kk]) / (rs[k] + rs[kk]) : rs[k];
            flujo += rc * (T[kk] - tc);
          }
        }
        const dif = (p.chi * flujo) / rs[k];
        const comp = (-tc * dpdT(rs[k], p.b, p.R)) / (rs[k] * p.cv) * div[k];
        salida[k] = adv + dif + comp;
    }
  }

  function pasoTemperatura(m) {
    const { N, solido, fuente, Tfuente, rho } = m;
    for (let k = 0; k < N; k++) m._rs[k] = solido[k] ? 0.1 : rho[k];
    gradiente(m, m.ux, m._gx, m._gy);
    for (let k = 0; k < N; k++) m._div[k] = m._gx[k];
    gradiente(m, m.uy, m._gx, m._gy);
    for (let k = 0; k < N; k++) m._div[k] += m._gy[k];

    for (let k = 0; k < N; k++) m._tmp[k] = fuente[k] ? Tfuente[k] : m.T[k];
    rellenar(m, m._tmp, m._T0);
    ladoDerecho(m, m._T0, m._k1);
    for (let k = 0; k < N; k++) m._tmp[k] = m._T0[k] + m._k1[k];
    rellenar(m, m._tmp, m._T1);
    ladoDerecho(m, m._T1, m._k2);
    for (let k = 0; k < N; k++) m.T[k] = m._T0[k] + 0.5 * (m._k1[k] + m._k2[k]);
  }

  function paso(m, n) {
    const total = n === undefined ? 1 : n;
    for (let s = 0; s < total; s++) {
      pasoFluido(m);
      if (!m.p.isotermico) pasoTemperatura(m);
      m.pasos++;
    }
  }

  // ------------------------------------------------------------ herramientas

  // desenfoque gaussiano separable, idéntico a scipy.ndimage.gaussian_filter
  // (sigma=1.2, truncate=4 → radio 5, mode="nearest")
  const RADIO_G = 5;
  const KERNEL_G = (function () {
    const s = 1.2, k = [];
    let tot = 0;
    for (let i = -RADIO_G; i <= RADIO_G; i++) { const w = Math.exp(-0.5 * (i * i) / (s * s)); k.push(w); tot += w; }
    return k.map((w) => w / tot);
  })();

  function suavizar(m, mascara) {
    const { nx, ny, N } = m;
    const a = new Float64Array(N), b = new Float64Array(N);
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        let s = 0.0;
        for (let j = -RADIO_G; j <= RADIO_G; j++) {
          const yy = Math.min(Math.max(y + j, 0), ny - 1);
          s += KERNEL_G[j + RADIO_G] * mascara[yy * nx + x];
        }
        a[y * nx + x] = s;
      }
    }
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        let s = 0.0;
        for (let j = -RADIO_G; j <= RADIO_G; j++) {
          const xx = Math.min(Math.max(x + j, 0), nx - 1);
          s += KERNEL_G[j + RADIO_G] * a[y * nx + xx];
        }
        b[y * nx + x] = s;
      }
    }
    return b;
  }

  function masaEn(m, k) {
    let r = 0.0;
    for (let i = 0; i < 9; i++) r += m.f[i * m.N + k];
    return r;
  }

  function agregarLiquido(m, mascara, u, v, Tred) {
    const p = m.p;
    const phi = suavizar(m, mascara);
    // cota de las celdas que el borde suavizado puede tocar
    for (let k = 0; k < m.N; k++) {
      if (m.solido[k] || phi[k] <= 0.02) continue;
      const rn = p.rhoVapor + (p.rhoLiquido - p.rhoVapor) * Math.min(Math.max(phi[k], 0), 1);
      const antes = masaEn(m, k);
      if (!(rn > antes + 1e-9)) continue;
      ponerEquilibrio(m, k, rn, u || 0, v || 0);
      m.masaAgregada += masaEn(m, k) - antes;
      if (Tred !== undefined && Tred !== null) m.T[k] = Tred * p.Tc;
    }
  }

  // entrada de velocidad: equilibrio de líquido fijado cada paso (Mundo.imponer_entrada)
  function imponerEntrada(m, mascara, u, v, Tred) {
    for (let k = 0; k < m.N; k++) {
      if (!mascara[k] || m.solido[k]) continue;
      const antes = masaEn(m, k);
      ponerEquilibrio(m, k, m.p.rhoLiquido, u || 0, v || 0);
      m.masaAgregada += masaEn(m, k) - antes;
      if (Tred !== undefined && Tred !== null) m.T[k] = Tred * m.p.Tc;
    }
  }

  function quitarLiquido(m, mascara) {
    for (let k = 0; k < m.N; k++) {
      if (!mascara[k] || m.solido[k]) continue;
      const antes = masaEn(m, k);
      ponerEquilibrio(m, k, m.p.rhoVapor, 0, 0);
      m.masaAgregada += masaEn(m, k) - antes;
    }
  }

  function agregarPared(m, mascara) {
    for (let k = 0; k < m.N; k++) if (mascara[k]) m.solido[k] = 1;
  }

  function quitarPared(m, mascara) {
    const { nx, ny } = m;
    for (let y = 1; y < ny - 1; y++) {
      for (let x = 1; x < nx - 1; x++) {
        const k = y * nx + x;
        if (!mascara[k] || !m.solido[k]) continue;
        m.solido[k] = 0; m.fuente[k] = 0; m.T[k] = m.p.T0;
        const antes = masaEn(m, k);
        ponerEquilibrio(m, k, m.p.rhoVapor, 0, 0);
        m.masaAgregada += masaEn(m, k) - antes;
      }
    }
  }

  function agregarFuente(m, mascara, Tred) {
    const T = Tred * m.p.Tc;
    for (let k = 0; k < m.N; k++) {
      if (!mascara[k]) continue;
      m.solido[k] = 1; m.fuente[k] = 1; m.Tfuente[k] = T; m.T[k] = T;
    }
  }

  // ------------------------------------------------------------ observables

  function observables(m) {
    const { N, nx, ny, rho, ux, uy, T, solido, p } = m;
    let masa = 0.0;
    for (let k = 0; k < 9 * N; k++) masa += m.f[k];
    let cin = 0, pot = 0, int = 0, Tliq = 0, nliq = 0;
    const umbral = 0.5 * (p.rhoLiquido + p.rhoVapor);
    for (let y = 0; y < ny; y++) {
      const h = ny - 1 - y;
      for (let x = 0; x < nx; x++) {
        const k = y * nx + x;
        if (solido[k]) continue;
        const r = rho[k];
        cin += 0.5 * r * (ux[k] * ux[k] + uy[k] * uy[k]);
        pot += r * p.gravedad * h;
        int += r * p.cv * T[k] - p.a * r * r;
        if (r > umbral) { Tliq += T[k]; nliq++; }
      }
    }
    return { masa, cinetica: cin, potencial: pot, interna: int, total: cin + pot + int, Tliq: nliq ? Tliq / nliq / p.Tc : NaN, pixelesLiquido: nliq };
  }

  const api = { crearMundo, paso, agregarLiquido, imponerEntrada, quitarLiquido, agregarPared, quitarPared, agregarFuente, observables, presion, ponerEquilibrio };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else raiz.Aguacero = api;
})(typeof self !== "undefined" ? self : this);
