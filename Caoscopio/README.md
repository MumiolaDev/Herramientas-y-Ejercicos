# Caoscopio

Un instrumento para *observar* sistemas dinámicos en el espacio de fases
— caóticos y no caóticos. Se especifica un sistema (péndulo simple,
péndulo doble, o péndulo forzado y amortiguado), una condición inicial y
unos parámetros numéricos con una API de Python tipada (dataclasses), y
la herramienta integra las ecuaciones de movimiento y entrega la
trayectoria lista para visualizar: la animación física (el péndulo
moviéndose), su retrato de fases, y — para el sistema forzado — el campo
vectorial y la sección de Poincaré que revelan sus atractores.

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

## Atractores: el péndulo forzado y amortiguado

Los dos péndulos de arriba son **conservativos**: la energía mecánica ni
se crea ni se destruye, y por el teorema de Liouville el volumen del
espacio de fases tampoco — dos trayectorias con condiciones iniciales
distintas nunca convergen entre sí. Eso significa que un sistema
Hamiltoniano, por más caótico que sea, **no puede tener un atractor**: no
hay ningún conjunto hacia el que "caigan" trayectorias arbitrarias.

`DrivenDampedPendulum` rompe esa conservación a propósito:
θ''+bθ'+(g/l)sinθ=A cos(ω_d t). El término -bθ' disipa energía, A cos(ω_d t)
la repone — la combinación contrae el volumen del espacio de fases, y *eso*
es lo que hace posible un atractor. No siendo conservativo, no tiene
`energia()` (ver el docstring de `systems/driven_pendulum.py`), así que la
validación es distinta:

- **Sin forzado, amplitud pequeña** (`test_decaimiento_subamortiguado_sin_forzado_coincide_con_solucion_analitica`):
  se reduce al oscilador armónico amortiguado clásico, con solución
  cerrada conocida — coincide a 2×10⁻⁵.
- **El atractor es independiente de la condición inicial**
  (`test_atractor_periodico_es_independiente_de_la_condicion_inicial`):
  con A=0.9, cuatro condiciones iniciales bien distintas convergen al
  *mismo* punto en la sección de Poincaré estroboscópica (una muestra por
  período de forzado) — verificado antes de escribir el test, no solo
  citado: las cuatro coinciden a 5 cifras significativas.
- **El régimen caótico no colapsa** (`test_atractor_caotico_no_colapsa_a_un_punto`):
  con A=1.15 (parámetro tomado de un ejemplo muy citado en la literatura
  de ecuaciones diferenciales: θ''+0.5θ'+sinθ=A cos(2t/3)), la sección de
  Poincaré tiene dispersión apreciable — un **atractor extraño**: acotado,
  con estructura fina, pero sin colapsar a un punto ni a una curva simple.

`examples/pendulo_forzado_atractor.py` dibuja ambos regímenes: el campo
vectorial congelado a la fase de forzado t≡0 (mod T_d — la misma fase en
la que se muestrea la sección de Poincaré, para que campo y puntos
correspondan al mismo instante del ciclo), y seis condiciones iniciales
"cayendo" hacia el atractor. En A=0.9 las seis espiralan hacia el mismo
punto; en A=1.15 llenan la misma región con estructura de capas — la
firma visual de un atractor extraño.

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
PYTHONPATH=. python3 examples/pendulo_forzado_atractor.py   # campo vectorial + atractores: ciclo límite vs. extraño
PYTHONPATH=. python3 examples/lorenz_mariposa.py             # 3D: la mariposa de Lorenz, sensibilidad extrema
```

(`PYTHONPATH=.` no es necesario si se instaló el paquete con `pip install -e .`)

Cada ejemplo deja su `.gif` (y, el de sensibilidad, además un `.png` de
|Δθ₁(t)|) en `salidas/` (no versionado), más el `print()` de conservación
de energía.

### Tests

```bash
pytest
```

## El atractor de Lorenz: el primer sistema en 3D, y el primero que no es un péndulo

`Lorenz` (ẋ=σ(y-x), ẏ=x(ρ-z)-y, ż=xy-βz) es una simplificación drástica
de la convección atmosférica que Lorenz publicó en 1963 — sin ángulo, sin
gravedad, el primer sistema del proyecto que no es un péndulo. Con los
parámetros clásicos (σ=10, ρ=28, β=8/3) es caótico y disipativo, y su
atractor extraño es la "mariposa" más reconocible de la teoría del caos
(el propio Lorenz acuñó la metáfora del "efecto mariposa" hablando de este
sistema, en una charla de 1972).

Dos validaciones independientes, ninguna apoyada en "hacia dónde parece
que converge":

- **Puntos fijos exactos** (`test_puntos_fijos_son_equilibrios_exactos`):
  para ρ>1 hay tres equilibrios cerrados en forma analítica —el origen y
  el par simétrico C±=(±√(β(ρ-1)), ±√(β(ρ-1)), ρ-1)— y evaluar
  `derivadas()` ahí debe dar cero exactamente. Es una prueba algebraica
  directa.
- **Contracción de volumen a la tasa exacta**
  (`test_tasa_de_contraccion_de_volumen_coincide_con_la_divergencia_teorica`):
  la divergencia del campo, ∇·f=-σ-1-β, es *constante* en todo el espacio
  de fases — así que un volumen infinitesimal se contrae exactamente como
  e^{(-σ-1-β)t}. Se verifica integrando un tetraedro de cuatro puntos
  infinitesimalmente cercanos y comparando la tasa de decaimiento de su
  volumen contra la fórmula: coinciden a 0.1%. Es el análogo, en un
  sistema disipativo, de "energía conservada" en uno Hamiltoniano — salvo
  que acá lo que se predice de antemano no es que algo se conserve, sino
  exactamente a qué tasa se contrae.

Un detalle honesto que vale la pena dejar anotado, no pulido:
`examples/lorenz_mariposa.py` grafica la separación entre dos
trayectorias con Δx₀=10⁻⁵. La expectativa naive —crecimiento exponencial
limpio desde t=0— no es lo que se observa: la separación se queda casi
plana (oscilando entre 10⁻⁵ y 2×10⁻⁵) durante los primeros ~12 segundos,
y solo despega con claridad después. No es un error: una perturbación
inicial arbitraria rara vez está alineada con la dirección de máxima
expansión del atractor (el "vector de Lyapunov" dominante) — hace falta
un tiempo de relajación para que la dinámica la realinee, antes de que la
tasa de crecimiento se estabilice en el exponente de Lyapunov máximo.
Es exactamente el tipo de matiz que este proyecto prefiere medir y
mostrar, no asumir.

## Roadmap

- **Exponente de Lyapunov**: cuantificar la tasa exponencial de
  divergencia que ya se ve en `examples/pendulo_doble_sensibilidad.py` y
  `examples/lorenz_mariposa.py` (hoy ambos solo grafican |Δ(t)|) — pide
  renormalizar periódicamente la separación entre las dos trayectorias y
  promediar su tasa de crecimiento, no solo medir la separación cruda (y,
  visto el matiz de arriba, descartar el tramo inicial de realineamiento
  antes de promediar).
- **Sección de Poincaré para el péndulo doble**: distinto del caso ya
  implementado (que muestrea estroboscópicamente a la frecuencia de un
  forzado externo) — para un sistema autónomo como el péndulo doble, la
  sección se define cortando la trayectoria cada vez que θ1 cruza un
  valor fijo (p. ej. θ1=0, ω1>0) y graficando (θ2,ω2) en esos instantes,
  a energía fija. Es la forma estándar de ver la estructura fina entre el
  régimen regular y el caótico en un sistema Hamiltoniano de 2 grados de
  libertad (toros KAM sobreviviendo, islas de regularidad en medio del
  mar caótico) — complementario al campo vectorial + atractor del péndulo
  forzado, que muestra la otra cara: qué pasa cuando el sistema deja de
  ser conservativo.
- **Más sistemas**: oscilador de Duffing (otro clásico con atractor
  extraño, doble pozo en vez de péndulo).
- **Integrador simpléctico** opcional, si la deriva de energía de RK4
  llega a ser un problema real para alguna trayectoria de interés (ver
  "El integrador" arriba).
- **Backend ASCII**, por consistencia con Colapsoscopio: un retrato de
  fase de baja resolución reproducible en terminal.
