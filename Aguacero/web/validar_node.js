// Puente para tests/test_web_vs_numpy.py: lee un estado de Mundo en JSON,
// lo avanza con web/nucleo.js y devuelve el resultado en JSON.
//   node web/validar_node.js entrada.json salida.json
const fs = require("fs");
const A = require("./nucleo.js");

const e = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const m = A.crearMundo(e.nx, e.ny, e.p);
m.f.set(e.f);
m.T.set(e.T);
m.solido.set(e.solido);
m.fuente.set(e.fuente);
m.Tfuente.set(e.Tfuente);
if (e.mascara_liquido) A.agregarLiquido(m, Uint8Array.from(e.mascara_liquido), e.ux || 0, e.uy || 0);
A.paso(m, e.pasos);
fs.writeFileSync(
  process.argv[3],
  JSON.stringify({ f: Array.from(m.f), T: Array.from(m.T), ux: Array.from(m.ux), uy: Array.from(m.uy), masaAgregada: m.masaAgregada })
);
