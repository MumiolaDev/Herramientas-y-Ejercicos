# Colapsoscopio

Un instrumento para *observar* la evolución temporal de ondas: se especifica
un medio (un potencial cuántico, por ahora), una condición inicial y unos
parámetros numéricos con una API de Python tipada (dataclasses), y la
herramienta integra la ecuación correspondiente y entrega la trayectoria
lista para visualizar — con matplotlib (imagen estática o animación .gif) o
directamente en la terminal, como "ASCII art" reproducible sin ninguna
dependencia gráfica.

El nombre juega con el "colapso" de la función de onda al medir, y el
sufijo "-scopio" de todo instrumento de observación. Aquí no colapsamos
nada de verdad — no hay medición, solo evolución unitaria — pero sí
observamos: |Psi|², Re/Im Psi, y los valores esperados <x>, <p>, <H> en
función del tiempo.

## Dos dominios físicos, no uno

El proyecto empezó siendo "la TDSE en 1D" y ahora cubre 1D + 2D cuánticos,
con un segundo dominio (ondas clásicas: acústica, Maxwell) planeado como
hermano — no como extensión del mismo código:

```
colapsoscopio/
├── quantum/           # ecuación de Schrödinger dependiente del tiempo (1D, 2D)
└── classical_waves/   # roadmap: ecuación de onda / Maxwell — FDTD, no split-operator
```

La razón de separarlos así, en vez de generalizar un único `Solver`, es
física antes que de ingeniería: la TDSE es de **primer orden en el tiempo**
y compleja — $i\hbar\,\partial_t\Psi = H\Psi$ —, y eso es justo lo que hace
funcionar al split-operator ($e^{-iHt/\hbar}$ es unitario, conserva
$\|\Psi\|^2$ exactamente). La ecuación de onda acústica
($\partial_t^2 u = c^2\nabla^2 u$) y Maxwell son de **segundo orden en el
tiempo**, reales (Maxwell además vectorial y acoplado E↔B): no existe un
operador de evolución unitario análogo que aplicar ahí, lo que se conserva
es energía del campo, no norma $L^2$ de una función de onda. El método
estándar en ese mundo es FDTD (malla de Yee, leapfrog), no split-step
Fourier. Forzar ambos bajo la misma abstracción rompería la que ya
funciona en vez de extenderla; comparten la *filosofía* (separar
malla/medio/condición-inicial/solver) pero no el código de bajo nivel.

Este documento describe `quantum/`, que es lo que existe hoy.

## Por qué así (y no una colección de scripts sueltos)

La versión "obvia" de este proyecto es un script por cada sistema: uno para
el pozo, otro para el oscilador, cada uno con su propia implementación de
Crank-Nicolson o Euler explícito y su propio `plt.show()`. Funciona para un
ejercicio, pero no escala: agregar un potencial nuevo implica reescribir el
integrador, y "eventualmente escalar a un átomo de hidrógeno" con ese
enfoque significa empezar de cero.

La apuesta de `quantum/` es separar el problema en cuatro piezas
ortogonales, cada una intercambiable sin tocar las demás (los mismos cuatro
nombres existen dos veces — sufijo `2D` — para el caso bidimensional):

1. **Malla** (`core/grid.py`, `core/grid2d.py`): dónde vive Psi y qué
   condición de frontera tiene.
2. **Potencial** (`potentials/`): qué es V(x) o V(x,y). Solo necesita saber
   evaluarse a sí mismo; opcionalmente sabe construir sus autoestados
   analíticos (1D).
3. **Condición inicial** (`initial_conditions/`): cómo es Psi en t=0.
4. **Solver** (`solvers/`): cómo se avanza Psi un paso dt.

`Simulation`/`Simulation2D` los junta y corre la evolución; el resultado
(`Trajectory`/`Trajectory2D`) es un array de numpy que no sabe nada de
matplotlib ni de terminales — los backends de `visualization/` son
consumidores de esa trayectoria, no partes del núcleo. Esta separación es
la que permite que "agregar un sistema nuevo" sea agregar una clase
`Potential`, no reescribir el integrador; y que "agregar hidrógeno" sea
agregar una malla radial y un potencial de Coulomb, reutilizando el mismo
solver conceptual.

## El método numérico: split-operator (split-step espectral)

Se implementa el método de Feit-Fleck-Steiger (1982), también llamado
split-step de Fourier o "split-operator": para dt pequeño,

```
e^{-i H dt/hbar} ≈ e^{-i V dt/(2hbar)} · e^{-i T dt/hbar} · e^{-i V dt/(2hbar)} + O(dt^3)
```

Cada factor es la exponencial de un operador diagonal (V en el espacio de
posiciones, T = p²/2m en el espacio donde T es diagonal), o sea un producto
elemento a elemento por una fase compleja — no se ensambla ni se invierte
ninguna matriz. Eso tiene dos consecuencias que son la razón de ser de este
diseño:

- **Unitariedad exacta**: la norma de Psi se conserva a precisión de
  máquina, no "aproximadamente" (ver `tests/test_conservacion.py` — el
  drift es ~1e-9, puro error de redondeo). No es una propiedad accesoria:
  es la garantía de que el solver no está inventando ni destruyendo
  probabilidad.
- **Se generaliza sin cambiar la idea**: todo lo que hace falta es una
  transformada donde T sea diagonal. Eso es exactamente lo que separa
  `Grid1D` en dos "sabores" de frontera:

  - `boundary="periodic"`: Transformada de Fourier (`np.fft`). Pensada
    para un dominio "abierto" (el oscilador armónico, una caja mucho más
    ancha que la extensión de Psi).
  - `boundary="dirichlet"`: Transformada seno discreta, DST-I
    (`scipy.fft.dst`). Sus autovalores de energía cinética son
    exactamente k_n = nπ/L — la cuantización que predice la teoría para
    un **pozo de potencial infinito real**, con Psi=0 en las paredes
    impuesto por la propia base espectral, no aproximado con una pared de
    potencial "muy alta pero finita".

  `Grid2D` (periódica en ambos ejes, FFT2) es la misma idea en dos
  dimensiones — ver la sección de la doble rendija más abajo — y escalar a
  un átomo de hidrógeno (roadmap) significa agregar una tercera base
  espectral —una malla radial con la transformada apropiada para el
  operador de Coulomb— implementando la misma interfaz
  `transformar_ida`/`transformar_vuelta`. El solver no cambia una línea.

## Validación

`potentials.base.Potential.autoestado()` da la forma cerrada de los
autoestados del pozo infinito (senos) y del oscilador armónico
(Hermite×gaussiana). Arrancar la simulación en un autoestado exacto (con la
condición inicial `Eigenstate`) da la validación más fuerte posible sin
depender de ningún otro cálculo numérico: un autoestado es estacionario,
así que |Psi(x,t)|² debe quedar *exactamente* igual a |Psi(x,0)|² en todo
instante, y Psi solo debe acumular la fase global e^{-iE_n t/ℏ}. Los tests
en `tests/test_autoestados.py` comparan justamente eso, y
`examples/validar_autoestados.py` lo corre a mano imprimiendo el error.

## Instalación

```bash
cd Colapsoscopio
pip install -r requirements.txt
# o, para desarrollo (deja el paquete importable en modo editable):
pip install -e .
```

## Uso

```python
from colapsoscopio import Grid1D, Simulation, SimulationConfig
from colapsoscopio.quantum.initial_conditions import GaussianPacket
from colapsoscopio.quantum.potentials import InfiniteWell
from colapsoscopio.quantum.visualization import AsciiAnimator, MatplotlibAnimator

grid = Grid1D(x_min=0.0, x_max=20.0, n_points=512, boundary="dirichlet")
config = SimulationConfig(
    grid=grid,
    potential=InfiniteWell(),
    initial_condition=GaussianPacket(x0=6.0, sigma=0.6, k0=6.0),
    dt=2e-4,
    n_steps=15000,
    guardar_cada=40,
)
traj = Simulation(config).run()

# Terminal, sin ninguna dependencia gráfica:
AsciiAnimator(traj).reproducir(fps=15)

# o como imagen/animación:
MatplotlibAnimator(traj).guardar("pozo.gif", fps=25)
```

`Grid1D`, `WaveFunction`, `Hamiltonian1D`, `Simulation`, `SimulationConfig`,
`Trajectory` y sus equivalentes `2D` se re-exportan desde `colapsoscopio`
directamente; todo lo demás (`potentials`, `initial_conditions`,
`solvers`, `visualization`) vive en `colapsoscopio.quantum.*`.

### Ejemplos ejecutables

```bash
PYTHONPATH=. python3 examples/pozo_infinito.py        # paquete rebotando en las paredes
PYTHONPATH=. python3 examples/oscilador_armonico.py   # paquete oscilando (estado casi-coherente)
PYTHONPATH=. python3 examples/barrera_potencial.py    # efecto túnel: <E> < V0 y aun así transmite
PYTHONPATH=. python3 examples/doble_rendija.py        # 2D: patrón de interferencia
PYTHONPATH=. python3 examples/billar_cuantico.py       # 2D: billar de Sinai, ergodicidad cuántica
PYTHONPATH=. python3 examples/validar_autoestados.py  # chequeo de estacionariedad
```

(`PYTHONPATH=.` no es necesario si se instaló el paquete con `pip install -e .`)

Cada ejemplo deja sus salidas en `salidas/` (no versionado): un `.gif` y un
`.txt` con la misma trayectoria en ASCII, más el `print()` de conservación
de norma y energía.

### Tests

```bash
pytest
```

## Efecto túnel (barrera de potencial)

`PotentialBarrier` (`potentials/barrier.py`) no tiene autoestados analíticos
—el problema natural ahí es de scattering, no de estados ligados en L²—, así
que la validación no es "la densidad queda estacionaria" sino algo más
físico todavía: con un paquete de energía media **menor** que la altura de
la barrera, debe aparecer probabilidad no despreciable del otro lado, algo
clásicamente prohibido de plano. `examples/barrera_potencial.py` corre
exactamente ese caso (`<E>=0.53` contra `V0=1.0`) e imprime tanto la
transmisión numérica del paquete real como el `T(E)` teórico de la barrera
rectangular *estacionaria* evaluado en esa energía media —quedan cerca
(0.20 numérico vs. 0.196 teórico) pero no iguales, y la razón es instructiva:
un paquete gaussiano no es monocromático, así que su transmisión real es un
promedio de `T(E)` sobre su propio espectro de energías, no `T(<E>)`.
`tests/test_barrera.py` fija ese comportamiento como regresión (transmisión
apreciable pero minoritaria, con la mayoría reflejada).

## 2D: la doble rendija

`Grid2D` + `Hamiltonian2D` + `SplitStepSolver2D` son la misma idea que la
versión 1D con FFT2 en vez de FFT — ninguna idea nueva, la prueba de que el
método no depende de la dimensión. `DoubleSlit` (`potentials/double_slit.py`)
es una pantalla opaca con dos aberturas; `examples/doble_rendija.py` manda
un paquete ancho en y (para iluminar ambas rendijas como un frente casi
plano) y el resultado, del otro lado, es el patrón de interferencia
clásico: un máximo central, un mínimo, un máximo secundario menor, no dos
manchas separadas como predeciría la intuición de "pasa por una rendija u
otra". `tests/test_2d.py` fija esa no-monotonía del perfil como regresión.

Dos trampas no obvias, ambas de las que conviene desconfiar en cualquier
simulación de difracción, no solo esta:

- **Rango dinámico de color**: la densidad transmitida/difractada es
  órdenes de magnitud más tenue que el pico del paquete incidente, todavía
  compacto — con una escala de color/intensidad *lineal* contra el máximo
  global, el patrón de interferencia (la parte interesante) queda
  invisible. Ambos backends 2D (`MatplotlibAnimator2D`, `AsciiAnimator2D`)
  comprimen el rango dinámico con `PowerNorm(gamma=0.4)` (el análogo del
  "stretch" no lineal que se usa para mostrar imágenes astronómicas de bajo
  brillo): sigue siendo una función monótona de la densidad real, solo que
  no lineal.

- **Campo cercano vs. campo lejano**: un patrón de doble rendija cortado
  *demasiado cerca* de la pantalla no muestra el patrón de interferencia de
  libro (varias franjas nítidas) sino una mancha con un único mínimo
  tenue. La escala que separa ambos regímenes es la **distancia de
  Rayleigh**, $L_R \sim d^2/\lambda$ (d = separación entre rendijas,
  λ = 2π/k₀ la longitud de onda de De Broglie del paquete): por debajo de
  $L_R$ se está en campo cercano (Fresnel), y hace falta propagar varias
  veces $L_R$ para entrar en campo lejano (Fraunhofer), donde el patrón
  sale limpio. Los parámetros de `examples/doble_rendija.py` (d=4, k₀=4,
  observado hasta x=25, con $L_R\approx10.2$) están calibrados para eso.

  Esto tiene una consecuencia de costo computacional real, no solo
  estética: observar en campo lejano exige un dominio más grande (más
  distancia que recorrer) y, para resolverlo con detalle, una malla más
  fina — y el costo del split-operator no escala linealmente con la
  resolución. El criterio de estabilidad exige $dt \sim dx^2$ (la fase
  cinética no puede aliasear), así que duplicar la resolución espacial
  no solo dobla los puntos de la malla: también *cuadruplica* el número de
  pasos necesarios para el mismo tiempo físico. Combinado con el costo
  $O(N\log N)$ de cada FFT2, el costo total escala aproximadamente como
  $t_{\text{total}} \cdot dx^{-4}\log(dx^{-2})$ — medido en este proyecto,
  pasar de una malla exploratoria (350×300) a la de alta fidelidad que usa
  `examples/doble_rendija.py` (875×750, campo lejano completo) es
  ~40× más caro (segundos vs. minutos), consistente con esa cuarta
  potencia. Es el precio de que el término cinético sea espectralmente
  exacto (sin error de truncamiento ahí): un esquema implícito
  (Crank-Nicolson, roadmap) no tendría ese límite en `dt`, a cambio de
  resolver un sistema lineal en cada paso en vez de una FFT.

## 2D: el billar cuántico (Sinai)

`Grid2D` gana el mismo segundo "sabor" de frontera que `Grid1D`:
`boundary="dirichlet"`, vía `scipy.fft.dstn`/`idstn` (DST-I en ambos ejes,
separable porque -∇² con Dirichlet en un rectángulo se diagonaliza en el
producto tensorial de las bases seno 1D de cada eje). `EmptyBilliard`
(billar rectangular vacío) es, otra vez, el único potencial 2D con
autoestados analíticos —producto de dos senos, uno por eje, exactamente
como el pozo infinito 1D pero separable en dos dimensiones— y por eso sirve
para la misma validación fuerte: arrancar en un autoestado (n,m) y
comprobar que |Psi|² queda estacionaria a precisión de máquina, con la
energía numérica exacta (ver `tests/test_billar.py`).

`SinaiBilliard` agrega un disco de potencial alto al centro del mismo
billar — el ejemplo canónico de billar caóticamente disperso: a diferencia
del rectángulo vacío (integrable, separable), la trayectoria clásica que
rebota contra el disco central es sensible a condiciones iniciales. Sin
autoestado analítico posible ahí, la validación es la de siempre
(conservación de norma/energía) más un chequeo físico directo: la densidad
dentro del disco debe quedar consistentemente despreciable, porque V ahí es
alto (`test_densidad_dentro_del_obstaculo_permanece_despreciable`).

`examples/billar_cuantico.py` lanza un paquete desde una esquina hacia el
disco. El resultado no es una "bolita" rebotando limpiamente varias veces:
tras chocar y difractarse alrededor del disco, y rebotar un par de veces
contra las paredes, la densidad desarrolla un patrón de **moteado
(speckle) irregular** que llena la cavidad de forma aproximadamente
uniforme — la firma cuántica de la ergodicidad clásica del billar de Sinai
(conjetura de Berry: los autoestados de alta energía de un sistema
clásicamente caótico se comportan localmente como una superposición de
ondas planas con fases aleatorias). Es un resultado más interesante que
forzar una apariencia de trayectoria clásica, y es lo que en realidad hace
interesantes a los billares cuánticos como objeto de estudio.

Una nota de rendimiento específica de este método: `dstn`/`idstn` usan
internamente una FFT de tamaño ~2(N+1) por eje, así que el tamaño de la
malla importa para el costo de una forma que no es solo "más puntos = más
lento" — un tamaño con `N+1` mal factorizado (p. ej. `N=510`, `N+1=511=7×73`)
midió ~57 ms/paso en este proyecto, contra ~25 ms/paso en un tamaño vecino
con `N+1=512=2⁹` y casi los mismos puntos. `examples/billar_cuantico.py`
usa `N=383` (`N+1=384=2⁷×3`) por esta razón, documentada ahí mismo.

## Roadmap

- **Más potenciales 1D**: pozo finito, doble pozo — cada uno es solo una
  clase `Potential` nueva; el solver no cambia.
- **Más potenciales/geometrías 2D**: billar de Bunimovich (estadio, otra
  geometría caótica clásica), rejilla de difracción de N rendijas.
- **Átomo de hidrógeno**: separar la parte angular (armónicos esféricos,
  exacta) de la radial u(r) = r·R(r), y resolver la TDSE radial con una
  base espectral adaptada al potencial de Coulomb (o una malla radial con
  Numerov/Crank-Nicolson si el split-operator no da un buen desempeño ahí).
- **`colapsoscopio.classical_waves/`**: ecuación de onda acústica y Maxwell
  vía FDTD (malla de Yee, leapfrog) — dominio hermano, no extensión del
  cuántico (ver la sección "Dos dominios físicos" más arriba para el porqué).
- **Solvers alternativos**: Crank-Nicolson como segunda implementación de
  `Solver` (1D), útil como referencia cruzada independiente del
  split-operator.
