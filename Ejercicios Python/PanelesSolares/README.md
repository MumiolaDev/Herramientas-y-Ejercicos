# Tarea Dev Junior - Ruuf

Postulante: Diego Passalacqua
Correo: dorlandopg@gmail.com
## 🛠️ Problema

El problema a resolver consiste en encontrar la máxima cantidad de rectángulos de dimensiones "a" y "b" (paneles solares) que caben dentro de un rectángulo de dimensiones "x" e "y" (techo).

### Ejecutar en Python
Para ejecutar la solución solo basta correr el archivo main.py utilizando python3
```bash
cd python
python3 main.py
```

## 📝 Solución
Para realizar este ejercicio primero tuve que considerar las restricciones que hacen mas simple el problema.
En primer lugar la forma del techo y los paneles es rectangular y en segundo, los paneles a colocar son todos del mismo tamaño. 

La estrategia fue definir el techo como un array de dimensiones dadas, donde el valor de la celda indica si el espacio
esta ocupado o no.

Se creó una funcion que coloqua los paneles dentro de este array si es posible y luego se utilizó fuerza bruta para
probar celda por celda si se puede poner el panel

La función que cuenta la cantidad de paneles intenta poner el panel con la orientacion que viene, si no puede 
intenta  rotando el panel 90 grados.

Un panel, al ser colocado actualiza el techo generado para que las celdas ocupadas ya no se encuentren
disponibles, eso provoca que pasemos por celdas que sabemos que estan ocupadas luego de poner un panel, para techos muy, muy grandes esto podria ser costoso.

### Bonus Implementado

Para ver una demostración del bonus los invito a igresar al jupyter notebook donde ustedes mismos pueden cargar sus imágenes


### Explicación del Bonus
Para completar el bonus no fue necesario cambiar el algoritmo, solo tuve que cambiar como se definen los techos.  En este caso
son imágenes que son transformadas a un array, esta vez se define tambien a -1 como indicador de que una celda no es techo disponible.

Al ser impagenes puedo probar con formas distintas rápidamente. La función funciona correctamente para encontrar una forma posible
de colocar los paneles pero no necesariamente la mejor.

Un triángulo puede ser inscrito con rectángulos y el algoritmo sigue igual


## 🤔 Supuestos y Decisiones

Para el bonus tuve que cambiar la entrada de la funcion calculate_panels para que en lugar de ancho y alto
se utilize una imagen con la forma del techo, de esta forma puedo probar diversas configuraciones y es muy fácil de visualizar.

Por otro lado, cuando el techo no es rectangular, al utilizar fuerza bruta no me aseguro a encontrar la máxima cantidad o bien, la forma mas óptima de colocar los paneles.
Para valores pequeños esto no deberia ser perceptible, aun no encuentro un caso donde pueda ver una mejor configuración pero a medida que la forma
del techo varia y la cantidad de paneles aumenta es mas probable de fallar

Sin embargo para efectos practicos funciona.

