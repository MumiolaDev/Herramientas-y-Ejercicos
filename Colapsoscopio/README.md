# Colapsoscopio

Un instrumento para *observar* la evolución temporal de funciones de onda:
se especifica un potencial, una condición inicial y unos parámetros
numéricos con una API de Python tipada (dataclasses), y la herramienta
integra la ecuación de Schrödinger dependiente del tiempo (TDSE) y entrega
la trayectoria Psi(x,t) lista para visualizar — con matplotlib (imagen
estática o animación .gif) o directamente en la terminal, como "ASCII art"
reproducible sin ninguna dependencia gráfica.

El nombre juega con el "colapso" de la función de onda al medir, y el
sufijo "-scopio" de todo instrumento de observación. Aquí no colapsamos
nada de verdad — no hay medición, solo evolución unitaria — pero sí
observamos: |Psi(x,t)|^2, Re/Im Psi, y los valores esperados <x>, <p>, <H>
en función del tiempo.

## Por qué así (y no una colección de scripts sueltos)

La versión "obvia" de este proyecto es un script por cada sistema: uno para
el pozo, otro para el oscilador, cada uno con su propia implementación de
Crank-Nicolson o Euler explícito y su propio `plt.show()`. Funciona para un
ejercicio, pero no escala: agregar un potencial nuevo implica reescribir el
integrador, y "eventualmente escalar a un átomo de hidrógeno" con ese
enfoque significa empezar de cero.

La apuesta de este proyecto es separar el problema en cuatro piezas
ortogonales, cada una intercambiable sin tocar las demás:

1. **Malla** (`core/grid.py`): dónde vive Psi y qué condición de frontera
   tiene.
2. **Potencial** (`potentials/`): qué es V(x). Solo necesita saber
   evaluarse a sí mismo; opcionalmente sabe construir sus autoestados
   analíticos.
3. **Condición inicial** (`initial_conditions/`): cómo es Psi(x,0).
4. **Solver** (`solvers/`): cómo se avanza Psi un paso dt.

`Simulation` los junta y corre la evolución; el resultado (`Trajectory`) es
un array de numpy que no sabe nada de matplotlib ni de terminales — los
backends de `visualization/` son consumidores de esa `Trajectory`, no
partes del núcleo. Esta separación es la que permite que "agregar un
sistema nuevo" sea agregar una clase `Potential`, no reescribir el
integrador; y que "agregar hidrógeno" sea agregar una malla 3D/radial y un
potencial de Coulomb, reutilizando el mismo solver conceptual.

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

  Escalar a un átomo de hidrógeno (roadmap) significa agregar una tercera
  base espectral —una malla radial con la transformada apropiada para el
  operador de Coulomb, o una malla 3D con FFT en las tres direcciones—
  implementando la misma interfaz `transformar_ida`/`transformar_vuelta`
  que ya usan `Hamiltonian1D` y `SplitStepSolver`. El solver no cambia una
  línea.

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
from colapsoscopio.initial_conditions import GaussianPacket
from colapsoscopio.potentials import InfiniteWell
from colapsoscopio.visualization import AsciiAnimator, MatplotlibAnimator

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

### Ejemplos ejecutables

```bash
PYTHONPATH=. python3 examples/pozo_infinito.py        # paquete rebotando en las paredes
PYTHONPATH=. python3 examples/oscilador_armonico.py   # paquete oscilando (estado casi-coherente)
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

## Roadmap

- **Más potenciales 1D**: pozo finito, barrera (efecto túnel), doble pozo
  — cada uno es solo una clase `Potential` nueva; el solver no cambia.
- **2D**: extender `Grid1D`/`Hamiltonian1D` a una malla 2D con FFT en ambos
  ejes — útil para ilustrar difracción, potenciales tipo billar cuántico.
- **Átomo de hidrógeno**: separar la parte angular (armónicos esféricos,
  exacta) de la radial u(r) = r·R(r), y resolver la TDSE radial con una
  base espectral adaptada al potencial de Coulomb (o una malla radial con
  Numerov/Crank-Nicolson si el split-operator no da un buen desempeño ahí).
  El resto de la arquitectura (`Simulation`, `Trajectory`, los backends de
  visualización) no debería necesitar cambios.
- **Solvers alternativos**: Crank-Nicolson como segunda implementación de
  `Solver`, útil como referencia cruzada independiente del split-operator.
- **Backend ASCII interactivo**: hoy `AsciiAnimator.reproducir()` ya anima
  en cualquier terminal; queda pendiente un modo con más de una fila
  (perfil 2D real en vez de una sola sparkline) para cuando existan
  visualizaciones 2D.
