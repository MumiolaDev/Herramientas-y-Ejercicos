# Caoscopio

Un instrumento para *observar* sistemas dinámicos en el espacio de fases
— caóticos y no caóticos. Se especifica un sistema (por ahora, péndulo
simple o péndulo doble), una condición inicial y unos parámetros
numéricos con una API de Python tipada (dataclasses), y la herramienta
integra las ecuaciones de movimiento y entrega la trayectoria lista para
visualizar: la animación física (el péndulo moviéndose) y su retrato de
fases, lado a lado.

Hermano conceptual de [`Colapsoscopio`](../Colapsoscopio) (mismo repo,
dominio distinto): allá se integra la ecuación de Schrödinger con un
propagador espectral exacto; acá se integran ecuaciones de movimiento
clásicas, típicamente no separables, con un integrador de propósito
general. La diferencia no es de preferencia — ver la sección
"El integrador" más abajo para el porqué exacto, y por qué eso cambia lo
que "validar el solver" significa de un proyecto al otro.

## Por qué así

Misma apuesta arquitectónica que Colapsoscopio: separar el problema en
piezas ortogonales en vez de escribir un script por sistema.

1. **Sistema** (`systems/`): un vector de estado y una regla
   dy/dt = f(t,y). Solo necesita eso; opcionalmente sabe calcular su
   propia energía mecánica, si es conservativo.
2. **Integrador** (`core/integrator.py`): cómo se avanza el estado un
   paso dt. Genérico, no sabe nada de péndulos.
3. **Condición inicial**: un array plano, sin envoltorio — no hace falta
   más ceremonia que esa acá.

`Simulation` los junta y corre la evolución; el resultado (`Trajectory`)
es un array de numpy que no sabe nada de matplotlib — `PendulumAnimator`
es un consumidor de esa trayectoria, no parte del núcleo. Agregar un
sistema nuevo (péndulo forzado, Duffing, Lorenz, tres cuerpos) es agregar
una clase con `derivadas()`, no tocar el integrador ni la visualización.

## El integrador: por qué no hay un análogo al split-operator acá

El split-operator de Colapsoscopio explota que H = T + V se separa en dos
piezas, cada una diagonal en una base distinta (posición para V, momento
para T): `e^{-iHdt/ℏ}` se factoriza en exponenciales *exactas*, cada una
una simple multiplicación por una fase — de ahí la unitariedad a
precisión de máquina, independiente de dt.

El Hamiltoniano de un péndulo doble no se separa así: la energía cinética
mezcla θ1' y θ2' con un factor cos(θ1−θ2) que depende de la propia
posición, así que no existe una base fija donde "la parte cinética" sea
diagonal de una vez y para siempre. Sin esa estructura, no hay un método
simple y de propósito general con conservación de energía *exacta* — de
ahí `RK4Integrator`: preciso (error local O(dt⁵)), pero sin ninguna
garantía algebraica. La energía se conserva *aproximadamente*, con una
deriva que depende de dt y se **mide**, no se declara.

(Los integradores simplécticos —Störmer-Verlet/leapfrog— sí garantizan
algo estructural para sistemas Hamiltonianos, pero exigen separar
cinética+potencial desacopladas, que el péndulo doble tampoco cumple sin
una transformación de coordenadas adicional. Queda en el roadmap si la
deriva de RK4 llega a ser un problema real para alguna trayectoria de
interés — hasta ahora, con dt=10⁻³, la deriva medida es de orden 10⁻⁹ en
20 s de simulación, ver más abajo.)

## Validación

Dos chequeos independientes, porque conservar energía por sí solo **no**
prueba que las ecuaciones de movimiento estén bien transcritas — solo que
son autoconsistentes (una `energia()` con el mismo error que
`derivadas()` seguiría "conservándose").

- **Conservación de energía** (`tests/test_double_pendulum.py`,
  `tests/test_simple_pendulum.py`): en los regímenes regular y caótico
  por separado, la deriva relativa medida es de orden 10⁻⁹ en 20
  segundos de simulación con dt=10⁻³.
- **El límite m2→0** (`test_limite_masa_nula_reduce_a_pendulo_simple`):
  con masa nula en el segundo péndulo, este no puede ejercer torque sobre
  el primero, así que θ1(t) del péndulo doble debe reducirse *exactamente*
  a un péndulo simple de longitud l1. Tomando ese límite a mano en las
  ecuaciones, el término de acoplamiento se cancela y dω1/dt →
  −(g/l1) sin θ1 — y en el código, corriendo ambos sistemas con la misma
  condición inicial en θ1, las trayectorias coinciden a 10⁻⁶. Es la
  prueba independiente de que las ecuaciones (no solo la energía) están
  bien.
- **Período del péndulo simple** (`test_periodo_converge_a_pequenas_oscilaciones`):
  para amplitud pequeña, el período numérico converge a T=2π√(l/g).

## Instalación

```bash
cd Caoscopio
pip install -r requirements.txt
# o, para desarrollo:
pip install -e .
```

## Uso

```python
import numpy as np
from caoscopio import DoublePendulum, Simulation, SimulationConfig
from caoscopio.visualization import PendulumAnimator

sistema = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
config = SimulationConfig(
    sistema=sistema,
    estado_inicial=np.array([2.4, 0.0, 2.4, 0.0]),  # theta1, omega1, theta2, omega2
    dt=1e-3,
    n_steps=20000,
    guardar_cada=60,
)
traj = Simulation(config).run()

PendulumAnimator(sistema, traj).guardar("pendulo.gif", fps=30)
```

### Ejemplos ejecutables

```bash
PYTHONPATH=. python3 examples/pendulo_simple.py             # no caótico: retrato de fase cerrado
PYTHONPATH=. python3 examples/pendulo_doble_regular.py      # caótico en potencia, energía baja: regular
PYTHONPATH=. python3 examples/pendulo_doble_caotico.py      # energía alta: caos de verdad
PYTHONPATH=. python3 examples/pendulo_doble_sensibilidad.py # dos condiciones iniciales casi iguales, divergiendo
```

(`PYTHONPATH=.` no es necesario si se instaló el paquete con `pip install -e .`)

Cada ejemplo deja su `.gif` (y, el de sensibilidad, además un `.png` de
|Δθ₁(t)|) en `salidas/` (no versionado), más el `print()` de conservación
de energía.

### Tests

```bash
pytest
```

## Roadmap

- **Exponente de Lyapunov**: cuantificar la tasa exponencial de
  divergencia que ya se ve en `examples/pendulo_doble_sensibilidad.py`
  (hoy solo grafica |Δθ₁(t)|) — pide renormalizar periódicamente la
  separación entre las dos trayectorias y promediar su tasa de
  crecimiento, no solo medir la separación cruda.
- **Sección de Poincaré**: cortar la trayectoria del péndulo doble cada
  vez que θ1 cruza un valor fijo y graficar (θ2,ω2) en esos instantes —
  la forma estándar de ver la estructura fina entre el régimen regular y
  el caótico (toros KAM sobreviviendo, islas de regularidad).
- **Más sistemas**: péndulo forzado y amortiguado (no conservativo — el
  primer sistema sin `energia()`), oscilador de Duffing, atractor de
  Lorenz (3D, requiere generalizar el panel físico o reemplazarlo por
  una proyección).
- **Integrador simpléctico** opcional, si la deriva de energía de RK4
  llega a ser un problema real para alguna trayectoria de interés (ver
  "El integrador" arriba).
- **Backend ASCII**, por consistencia con Colapsoscopio: un retrato de
  fase de baja resolución reproducible en terminal.
