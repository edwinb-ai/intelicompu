import numpy as np


class Poblacion:
    """Posibles soluciones del óptimo de una función.

    Esta clase crea posibles soluciones para una función a optimizar. Toma la dimensión
    del espacio, los límites de búsqueda y el número de elementos a tomar en cuenta.

    Attributes:
        dimension: Un entero que determina la dimensión de la función y el espacio.
        lim: Una lista [float, float] que contiene los límites de cada dimensión para restringir la búsqueda,
            el primer float es el mínimo, el segundo el máximo.
        elementos: Un entero que determina el número de posibles soluciones.
        valores: Un arreglo numpy de flotantes de tamanño (elementos, dimension) que contiene
            muestras de una distribución uniforme determinado por lim.
    """
    def __init__(self, dim, limites, total_individuos):
        """Se inicializa la clase con la dimensión, límites y el total de muestras
        a agregar. Siempre se debe especificar la dimensión, los límites y el número
        de elementos en la población.
        
        Los valores se inicializan a None porque se asignan posteriormente.
        """
        self.dimension = dim
        self.lim = limites
        self.elementos = total_individuos
        self.valores = None

    def inicializar(self):
        """Se crear un arreglo de numpy de flotantes de tamaño (elementos, dimension) con valores
        muestreados de una distribución uniforme, después se asignan a valores. Esta es la población
        asignada.

        """
        self.valores = np.random.uniform(
            *self.lim, size=(self.elementos, self.dimension)
        )

    @property
    def puntos(self):
        """Propiedad que devuelve el valor de valores."""
        return self.valores


class Optimizacion:
    """Implementación del algoritmo de optimización UMDA (Univariate Marginal Distribution Algorithm)
    que encuentra el mínimo de una función objetivo mediante el muestreo de una distribución normal.

    Attributes:
        objetivo: Una función de Python que corresponde a la función objetivo.
        dimension: Valor entero que determina la dimensión de la función.
        lim: Una lista [float, float] que contiene los límites de cada dimensión para restringir la búsqueda,
            el primer float es el mínimo, el segundo el máximo.
        elementos: Un entero que determina el número de posibles soluciones.
        mejores: Un entero constante que corresponde a la tercera parte de elementos.
        pasos: Un entero que contiene la información del número de iteraciones, el default es 100.
        poblacion_valores: Variable donde se guardará la información de las posibles soluciones.
        evaluaciones: Variables que guarda la información de evaluar la función en todos los elementos 
            que pertenecen a a poblacion_valores.
    
    """
    def __init__(self, func, dim, limites, poblacion, iteraciones=100):
        """Se inicializan todos los atributos excepto poblacion_valores y evaluaciones
        que cambian de acuerdo a la ejecución del código.
        """
        self.objetivo = func
        self.dimension = dim
        self.lim = limites
        self.elementos = poblacion
        self.mejores = self.elementos // 3
        self.pasos = iteraciones
        self.poblacion_valores = None
        self.evaluaciones = None

    def actualizar(self):
        """Crea un arreglo vacío donde se guardan los valores de poblacion_valores y sus evaluaciones
        con objetivo. El arreglo temp_arreglo tiene dimensión (elementos, dimension + 1) para guardar
        toda la población y además las evaluaciones en la última columna.
        """
        temp_arreglo = np.zeros((self.elementos, self.dimension + 1))
        temp_arreglo[:, :-1] = self.poblacion_valores
        temp_arreglo[:, -1] = np.array(
            [self.objetivo(i) for i in self.poblacion_valores]
        )
        # copiar el arreglo creado para evitar aliasing
        self.evaluaciones = np.copy(temp_arreglo)

    def optimizar(self):
        """Implementación del algoritmo UMDA. Se toman los valores de poblacion_valores, se seleccionan
        los elementos hasta el valor de mejores, después se realiza una evaluación de objetivo y se ordenan
        de forma descendiente, de mejor candidato a peor. Una vez ordenados se crea una nueva población
        encontrando la media y desviación estándar por cada dimensión y se muestrea de una distribución
        normal de tal forma que en cada dimensión se crea una distribución normal hasta convergencia.
        """
        poblacion = Poblacion(self.dimension, self.lim, self.elementos)
        poblacion.inicializar()
        self.poblacion_valores = poblacion.puntos
        # crear un arreglo para los q mejores
        q_mejores = np.zeros((self.mejores, self.dimension + 1))

        for _ in range(self.pasos):
            # siempre actualizar los valores
            self.actualizar()
            # ordenar los puntos dado el valor del objetivo, de mejor a peor
            self.evaluaciones = self.evaluaciones[self.evaluaciones[:, -1].argsort()]
            # escoger los q mejores
            q_mejores = self.evaluaciones[: self.mejores, :]
            # se toma el arreglo transpuesto para iterar sobre dimensión y no elementos
            for i in q_mejores[:, :-1].T:
                self.poblacion_valores = np.random.normal(
                    i.mean(), i.std(), size=self.poblacion_valores.shape
                )

    @property
    def resultado(self):
        """Propiedad que devuelve el primer valor de evaluaciones, que corresponde al valor
        mínimo, el mejor resultado.
        """
        return self.evaluaciones[0, :]


if __name__ == "__main__":

    # Resolver el problema de la esfera
    def esfera(x):
        # considerando que x es un arreglo de numpy
        return sum((x - 2.0) ** 2)

    # instanciar al optimizador
    optim_esfera = Optimizacion(esfera, 50, [-5.0, 10.0], 1000)
    optim_esfera.optimizar()
    print("Esfera")
    print("Resultado: {}".format(optim_esfera.resultado[:-1]))
    print("Valor mínimo: {}".format(optim_esfera.resultado[-1]))

    # Resolver la función Rastrigin
    def rastrigin(x):
        # Mínimo de 0 en (0, ..., 0)
        # considerando que x es un arreglo de numpy
        return 10.0 * len(x) + sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x))

    # instanciar al optimizador
    optim_trid = Optimizacion(rastrigin, 10, [-5.12, 5.12], 1000)
    optim_trid.optimizar()
    print("Rastrigin")
    print("Resultado: {}".format(optim_trid.resultado[:-1]))
    print("Valor mínimo: {}".format(optim_trid.resultado[-1]))
