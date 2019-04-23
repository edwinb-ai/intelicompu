---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# _Particle Swarm Optimizer_ (PSO)

PSO es un método de optimización inspirado en el patrón de comportamiento de parvadas de aves y cardúmenes de peces. Imagina una parvada de palomas circulando al rededor de un área en donde pueden oler comida. La paloma que se encuentre más cerca de la fuente de comida será aquella que zurreé más fuerte. Todas las demás palomas se apresurarán en irse hacia donde se encuentra ést. Si cualquier otra ave se acerca más a la fuente de comida, entonces esta chillará más fuerte y todas se apresurarán en irse hacia donde se encuentra ésta. El evento continuará ocurriendo y las aves irán agrupandose al rededor de la fuente de comida, cada vez más y más cerca, hasta que alguna encuentre el alimento.

Las primeras ideas fueron propuestas por [Kennedy y Eberhart (1995)](https://ieeexplore.ieee.org/document/494215) al igual que las primeras simulaciones ([1995](https://ieeexplore.ieee.org/document/488968), 1996).

El algoritmo implementado en el programa realizado en el presente trabajo se obtuvo de *Riccardo Pili, James Kennedy y Tim Blackwell, _"Particle swarm optimization. An overview._"* ([2007](https://link.springer.com/article/10.1007/s11721-007-0002-0)). En su trabajo la evolución del algoritmo, desde su introducción en 1995, es discutida. En él se presentan algunas versiones y aspectos del algoritmo, así como el efecto de los parámetros utlizados. **Si se desea profundizar más sobre el PSO, así como los distintos métodos y cuando utilizarlos, el origen y significado de las ecuaciones y parámetros utilizados, etc; favor de acudir al artículo mencionado así como a sus referencias.**

## Panorama general

En PSO, $n$ número de entidades (partículas) son colocadas en el _espacio de búsqueda_<sup>[1](#espacio)</sup> de la _función objetivo_<sup>[2](#func)</sup>. Cada individuo está compuesto por vectores $D$-dimensionales, en donde la dimensión está dada por la dimensionalidad del espacio de búsqueda. Éstos contienen la información de de la posición actual $\textbf{x}_i$, la mejor posición anterior<sup>[3](#optimizar)</sup> $\textbf{p}_i$ y la velocidad $\textbf{v}_i$, de la $i$-ésima partícula. **La mejor posición anterior $\textbf{p}_i$, es aquella que pertenece a la de la i-ésima partícula con el mejor valor calculado hasta ahora de la función objetivo $pbest_i$**.
<br>En cada iteración la posición actual $\textbf{x}_i$ es evaluada en la función objetivo. Si la posición actual es mejor que $\textbf{p}_i$, entonces guardamos o actualizamos $\textbf{p}_i$ con la posición $\textbf{x}_i$ y el valor de la función objetivo evaluada en dicha posición en $pbest_i$.

### ¿Es necesario guardar la mejor posición $\textbf{p}_i$? ¿No basta con simplemente guardar su valor $pbest_i$?

Retomando la introducción, las partículas se deben de ir agrupando como se agrupan las aves, posicionandose hacia donde se encuentra aquella que chilla más, o mejor dicho, hacia la posición $\textbf{p}_i$. Podemos intuír que las posiciones de todas las partículas en el ejambre deben ser actualizadas en dirección de $\textbf{p}_i$ , de manera tal que la partículas tomen valores al rededor de esa posición y ésta sea afinada a través de cada iteración. 

Es muy importante considerar que existen funciones que cuentan una gran cantidad de mínimos locales y que además se complican con la dimensionalidad. Por ello es que muchas partículas son arrojadas para explorar el espacio, en vez de una sola; de manera que se puede evitar, en lo mejor de lo posible, permanecer en mínimos locales . Si alguna partícula cae en un mínimo, entonces todas las demás se moveran hacia éste en base a $\textbf{p}_i$. Si alguna otra cae en otro mínimo local, menor al anterior, entonces comenzarán a dirigirse hacia una posición con un mejor $pbest_i$. Ésto ocurrirá iterativamente hasta que se cumpla algún criterio<sup>[4](#criterios)</sup> y con suerte se encuentre el mínimo global, o una buena aproximación de éste.

El siguiente algoritmo es una adaptación del agoritmo original, en el cual únicamente se modifica la manera en la que originalmente es calculada la velocidad. Éste fue desarrollado por Clerk y Kennedy ([2002](https://ieeexplore.ieee.org/document/985692)). Clerk y Kennedy incluyeron _coeficientes de constricción_ para controlar la convergencia del algoritmo dado que desde un principio se notaron casos en los cuales el calculo de las velocidades comenzaba hacer que las partículas comenzaran a diverger. Para ello originalmente se introduce un parámetro de velocidad máxima que impida que la velocidad crezca más de $V_{max}$.

El cálculo **original** de la velocidad está dado por:

$$\textbf{v}_i \leftarrow \textbf{v}_i + \textbf{U}(0,\phi_1) \otimes (\textbf{p}_i-\textbf{x}_i) +  \textbf{U}(0,\phi_2) \otimes (\textbf{p}_g-\textbf{x}_i) $$

En donde $\textbf{U}(0,\phi _i)$ representa un vector de números aleatorios uniformemente distribuidos en $[0,\phi]$, generado de manera aleatoria en cada iteración y para cada partícula.

Utilizando los _coeficientes de constricción_, la expresión anterior se convierte en:

$$\textbf{v}_i \leftarrow \chi (\textbf{v}_i + \textbf{U}(0,\phi_1) \otimes (\textbf{p}_i-\textbf{x}_i) +  \textbf{U}(0,\phi_2) \otimes (\textbf{p}_g-\textbf{x}_i) ) $$

En donde:

$$ \chi = \frac{2}{\phi - 2 + (\phi^2-4\phi)^{1/2}} $$

y $\phi=\phi_1+\phi_2$.

Usualmente se utilizan valores de $\phi = 4.1$ con $\phi_1=\phi_2$. Esta constricción de Clerc finalmente hará que las partículas converjan sin necesidad de controlar una $V_{max}$.

Otro método, y quizás el más comúnmente utilizado para controlar la busqueda reduciendo la importancia del parámetro $V_{max}$, es introducir un término llamado _peso inercial_ $w$ (Shi and Eberhart [1998](https://ieeexplore.ieee.org/document/699146)), de la siguiente manera:

$$\textbf{v}_i \leftarrow w\textbf{v}_i + \textbf{U}(0,\phi_1) \otimes (\textbf{p}_i-\textbf{x}_i) +  \textbf{U}(0,\phi_2) \otimes (\textbf{p}_g-\textbf{x}_i) $$

En el presente algoritmo se utiliza la constricción de Clerc, pero se mapea al uso del peso inercial mediante $w \leftrightarrow \chi$. Esto es:

* $w = 0.7298$
* $\phi_1 = \phi_2 = 1.49618$

## Algoritmo

1. *Inicializar la población de partíulas*.

* *Ciclo*
  * **Evaluar** cada partícula en la función objetivo.
  * **Comparar** el mejor valor de la función objetivo, con el de todas las partículas.
  * **Guardar o actualizar** el mejor valor encontrado de la función objetivo y su respectiva posición en el espacio.
  * **Actualizar** posición y velocidad de cada partícula:

    \begin{cases} 
    \text{$\textbf{v}_i$} &\leftarrow\quad\text{$\textbf{v}_i + \textbf{U}(0,\phi_1) \otimes (\textbf{p}_i-\textbf{x}_i) +  \textbf{U}(0,\phi_2) \otimes (\textbf{p}_g-\textbf{x}_i) $}, & (1) \\
    \text{$\textbf{x}_i$} &\leftarrow\quad\text{$\textbf{x}_i+\textbf{v}_i$} & (2)
    \end{cases}

En donde $\textbf{U}(0,\phi _i)$ representa un vector de números aleatorios uniformemente distribuidos en $[0,\phi]$, generado de manera aleatoria en cada iteración y para cada partícula.

<br><div><p><a name="espacio"><sup>1</sup></a><sub> El espacio de busqueda hace referencia al dominio de la misma</sub></p>
<p><a name="func"><sup>2</sup></a><sub> La función objetivo es aquella que se busca optmimizar</sub></p>
<p><a name="optimizar"><sup>3</sup></a><sub> Comúnmente en PSO lo que se busca es minimizar la función objetivo.</sub></p>
<p><a name="criterios"><sup>4</sup></a><sub> Usualmente de error y/o un número máximo de iteraciones.</sub></p></div>

<br>
<br>
<br>
<!-- #endregion -->

## Implementación en python

Para la implementación y prueba del código se utilizarán dos funciones prueba:

1. [Rastrigin](https://www.sfu.ca/~ssurjano/rastr.html) (3 dimensiones)
2. [Goldstein-Price](https://www.sfu.ca/~ssurjano/goldpr.html) (2 dimensiones)

y se utilizará lo mejor de lo posible la misma notación introducida anteriormente para los parámetros y las variables con el objetivo de facilitar el seguimiento del artículo anteriormente mencionado.

### 1. Función de Rastrigin

Se importan primeramente las librerías necesarias

```python
import numpy as np
from numpy import random as rnd
import math
```

Se declaran algunas variables globales

```python
D = 3           # número de dimensiones
npart = 50      # número de partículas
nepochs = 100   # máximas iteraciones
w = 0.7298      # inertia weight 
phi = 1.49618   # acceleration coefficients or stiffness:
```

Se definen algunas constantes para simplificar los cálculos de la función de Rastrigin

```python
k = 2.0 * np.pi
A = 10
An = A * D
# condiciones de frontera
a = -5.12
b = 5.12
```

Se define la función

$$ f(\textbf{x}) = AD + \sum^D_{i=1}[x_i^2 - Acos(2\pi x_i)] $$

en donde $A = 10$ con $-5.12\leq x_i \leq 5.12$

```python
def evaluate(x):
    t1 = np.square(x)
    t2 = A * np.cos(k * x)
    res = An + np.sum(t1 - t2)
    return res
```

Se crea una clase para contener la información de los vectores correspondientes a cada partícula, como un objeto de la clase.

```python
class Particle:
    def __init__(self):
        # vector de posición
        self.x = np.array(D)
        # vector de velocidad
        self.v = np.array(D)

        # posiciones y velocidades iniciales aleatorias
        for i in range(D):
            self.x = rnd.uniform(a,b,D)
            self.v = rnd.uniform(a,b,D)

        # mejor posición anterior inicial es la actual 
        self.p = np.copy(self.x)
        # mejor valor de la partícula es el actual
        self.val = evaluate(self.p)

    def compute_velocity(self, pg):
        t1 = w * self.v
        U1 = rnd.uniform(0,phi,D)
        t2 = U1 * (self.p - self.x)
        U2 = rnd.uniform(0,phi,D)
        t3 = U2 * (pg - self.x)
        self.v = t1 + t2 + t3

    def compute_position(self):
        self.x += self.v

    def update_pbests(self):
        self.p = np.copy(self.x)
        self.val = evaluate(self.p)
```

La siguiente es una función para manejar todas las acciones del PSO

```python
def PSO():
    # ----------------------------------
    # INICIALIZACION

    # lista para albergar el enjambre
    swarm = []
    # arreglo para la mejor posición en el enjambre
    pg = np.empty(D)
    # llenar el enjambre de partículas
    for i in range(npart):
        swarm.append(Particle())

        # designar a la primer partícula como la mejor en el enjambre
        if i == 0:
            # mejores valores en el enjambre
            pg = np.copy(swarm[i].x)
            sbest = swarm[i].val

        # comparar con la partícula anterior
        if i > 0 and swarm[i].val < sbest:
            # si la partícula actual es mejor que la anterior
            sbest = swarm[i].val
            pg = np.copy(swarm[i].x)

    # -------------------------------
    # CICLO

    epoch = 0
    while epoch < nepochs:
        for i in range(npart):
            # Computar nueva velocidad
            swarm[i].compute_velocity(pg)

            # Computar nueva posición
            swarm[i].compute_position()

            # evaluar en la función objetivo 
            swarm[i].val = evaluate(swarm[i].x)

            # actualizar los mejores valores del enjambre hasta ahora
            swarm[i].update_pbests()
            if swarm[i].val < sbest:
                pg = np.copy(swarm[i].x)
                sbest = swarm[i].val

        epoch += 1

    return pg, sbest
```

Se sabe que la función de Rastrigin tiene como valor mínimo:

$$f(0,0,0) = 0$$

De hecho para cualquier dimensión los mínimos globales de la función y su valor en dicha posición son $\textbf{x}=0$ y $f(\textbf{x})=0$, respectivamente.

```python
optimal = PSO()
np.set_printoptions(suppress=True)

print("\n\noptimal solution found at:\n{0}\n\n"
        "With value:\t{1}\n\n".format(optimal[0], optimal[1]))
```

### 2. Función de Goldstein-Price

Ahora se define la función de Goldstein-Price

$$ f(x,y) = [ 1 + (x + y + 1)^2 (19 - 14x + 3x^2 - 14y + 6xy + 3y^2)]
[30 + (2x-3y)^2 (18 - 32 + 12x^2 + 48y 0 36xy + 27y^2)]$$

en donde se sabe que el mínimo global se encuentra en

$$ f(0,-1) = 3 $$

dentro del dominio $ -2 \leq x,y \leq 2 $

```python
def evaluate(x):
    t1 = 1 + np.sum(x)
    t1 *= t1
    t2 = 19 - 14*x[0] + 3*x[0]*x[0] - 14*x[1] + 6*x[0]*x[1] + 3*x[1]*x[1]
    t3 = (2*x[0] - 3*x[1])**2
    t4 = 18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2
    res = (1 + t1*t2) * (30 + t3*t4)

    return res
```

```python
D = 2           # número de dimensiones
npart = 50      # número de partículas
nepochs = 100   # máximas iteraciones
w = 0.7298      # inertia weight 
phi = 1.49618   # acceleration coefficients or stiffness:
# dominio al que se restringe la función
a = -2.0
b = 2.0
```

```python
optimal = PSO()
np.set_printoptions(suppress=True)

print("\n\noptimal solution found at:\n{0}\n\n"
        "With value:\t{1}\n\n".format(optimal[0], optimal[1]))
```
