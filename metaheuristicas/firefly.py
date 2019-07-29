import numpy as np
import copy
import operator


class Firefly:
    """
    Esta es la clase base Firefly que crea vectores de posición y valor de una función
    en esta posición.

    Atrributes:
        intensidad: Arreglo numpy de valores flotantes que corresponde a la función
            objetivo evaluada en el arreglo posicion
        posicion: Arreglo numpy de valores flotantes que corresponde a la posición dentro del espacio
            de solución
        lim_sup: Valor flotante que corresponde al límite superior que puede tomar el arreglo de posicion
        lim_inf: Valor flotante que corresponde al límite superior que puede tomar el arreglo de posicion
        dim_fire: Valor flotante que corresponde a la dimensión del espacio, esto es, el
            tamaño del arreglo multidimensional
    """

    def __init__(self, sup, inf, dim):
        """Inicializa todos los atributos a None o utilizando los valores que
        se introduzcan cuando se inicializa.

        Args:
            sup: Valor flotante que corresponde al límite superior del espacio de búsqueda
            inf: Valor flotante que corresponde al límite inferior del espacio de búsqueda
            dim: Valor flotante que corresponde al tamaño del arreglo numpy
        """
        self.intensidad = None
        self.posicion = None
        self.lim_sup = sup
        self.lim_inf = inf
        self.dim_fire = dim

    @property
    def pos(self):
        """
        Devuelve el valor de la posición.

        Returns:
            posicion: Arreglo numpy de valores flotantes
        """
        return self.posicion

    @pos.setter
    def pos(self, val):
        """
        Asigna el valor val al valor de la posición.

        Args:
            val: Arreglo numpy de valores flotantes
        """
        self.posicion = val

    @property
    def brillo(self):
        """
        Devuelve el valor de la función (brillo).

        Returns:
            intensidad: Valor flotante que corresponde a evaluar la función
                objetivo en el arreglo self.posicion
        """
        return self.intensidad

    @brillo.setter
    def brillo(self, val):
        """
        Asigna el valor val al valor del brillo.

        Args:
            val: Valor flotante
        """
        self.intensidad = val

    def ini_posicion(self):
        """
        Genera una posición aleatoria utilizando la dimensión del espacio y los límites
        impuestos desde la construcción de la clase.
        """
        posicion_aleatoria = (self.lim_sup - self.lim_inf) * np.random.random_sample(
            self.dim_fire
        ) + self.lim_inf
        self.posicion = posicion_aleatoria


class FAOpt:
    """
    Esta es la implementación del método Firefly Algorithm de Yang & He (2013) para
    optimización multiobjetivo de funciones. Se emplea un método especial para ajustar
    un parámetro y acelerar la convergencia. Este método acepta una función, una dimensión,
    un rango de valores donde buscar la solución. Para los parámetros del método se
    dejan por default los originales propuestos por el autor.

    Attributes:
        alpha_opt: Valor flotante que corresponde al parámetro alfa
        beta_opt: Valor flotante que corresponde al parámetro beta
        gamma_opt: Valor flotante que corresponde al parámetro gamma
        lim_sup: Valor flotante para el máximo valor de búsqueda del espacio de solución
        lim_min: Valor flotante para el mínimo valor de búsqueda del espacio de solución
        func_opt: Función de Python que corresponde a la función objetivo
        f_args: Tupla de valores en caso que func_opt tenga argumentos adicionales
        dim_opt: Valor flotante que corresponde a la dimensión del espacio búsqueda
        pob_tam: Valor entero donde se guarda el número de elementos de la población
        poblacion: Lista de Python de objetos tipo Firefly
    """

    def __init__(
        self, func, dim, tam_pob, inf, sup, alpha=0.9, beta=0.2, gamma=1.0, args=()
    ):
        """
        Se inicializan todos los valores y se genera una lista de población instanciando
        la clase Firefly según el número de elementos en la población.

        Args:
            func: Cualquier tipo de objeto llamable de Python, normalmente una función que implemente
                la función objetivo a minimizar
            dim: Valor flotante que corresponde a la dimensión del espacio de búsqueda.
            tam_pob: Valor entero que corresponde al número de elementos que buscarán en el espacio
                de solución
            inf: Valor flotante para el límite inferior del espacio de soluciones
            sup: Valor flotante para el límite superior del espacio de soluciones
            alpha: Valor flotante que modifica el parámetro alfa del método
            beta: Valor flotante que modifica el parámetro beta del método
            gamma: Valor flotante que modifica el parámetro gamma del método
            args: Una tupla de Python para incluir argumentos adicionales de la función func
        """
        self.alpha_opt = alpha
        self.beta_opt = beta
        self.gamma_opt = gamma
        self.lim_sup = sup
        self.lim_inf = inf
        self.func_opt = func
        self.f_args = args
        self.dim_opt = dim
        self.pob_tam = tam_pob
        #  Para crear la población total de posibles soluciones (fireflies) se crea
        # una lista de objetos Firefly con el tamaño especificado por el usuario
        self.poblacion = [Firefly(sup, inf, dim) for i in range(tam_pob)]

    def ini_poblacion(self):
        """
        Se inicializa la posición y brillo de la población de forma aleatoria.
        """
        #  Por cada firefly en la población
        for i in self.poblacion:
            # Asignarle una posición aleatoria
            i.ini_posicion()

    def movimiento(self):
        """
        Este método implementa el algoritmo completamente, mueve las partículas
        según el algoritmo propuesto por el autor.
        """
        # Copiar la población anterior
        tmp_poblacion = copy.copy(self.poblacion)

        #  Escala del sistema
        escala = abs(self.lim_sup - self.lim_inf)

        # Iterar entre pares de fireflies, utilizando iteradores de Python
        for i in self.poblacion:
            #  Se necesita el índice de la población anterior actual para cambiar
            #  el valor correspondiente de la población nueva
            for j, k in enumerate(tmp_poblacion):
                # Actualizar la distancia según el algoritmo
                dist = np.power(np.linalg.norm(i.pos - k.pos), 2.0)
                # Si sucede que es una mejor solución
                if i.brillo > k.brillo:
                    #  Implementar el Firefly Algorithm
                    beta_new = (1.0 - self.beta_opt) * np.exp(-self.gamma_opt * dist) + self.beta_opt
                    estocastico = (
                        self.alpha_opt
                        * (np.random.random_sample(self.dim_opt) - 0.5)
                        * escala
                    )
                    tmp_pos = i.pos * (1.0 - beta_new) + k.pos * beta_new + estocastico

                    # Verificar que las posiciones no se salgan de los límites
                    np.clip(tmp_pos, self.lim_inf, self.lim_sup, out=tmp_pos)

                    #  Actualizar la posición
                    self.poblacion[j].pos = tmp_pos
                    #  Actualizar el brillo de esa posición
                    self.poblacion[j].brillo = -self.func_opt(
                        self.poblacion[j].pos, *self.f_args
                    )

    def optimizar(self, max_gen, optim=False):
        """
        Esta función implementa el movimiento de las partículas en la población durante un
        número fijo de ciclos.
        
        Args:
            max_gen: Valor entero para el número de pasos a ejecutar el movimiento de las
                partículas
            optim: Valor booleano, cuando es True se devuelve el mínimo y la función
                objetivo evaluada en ese mínimo

        Returns:
            Una tupla de arreglo numpy y valor flotante o un arreglo numpy, según sea
            el valo de optim.
        """
        # Inicializar la población, posicion y brillo iniciales
        self.ini_poblacion()
        for i in self.poblacion:
            i.brillo = -self.func_opt(i.pos, *self.f_args)

        # Mover las fireflies tanto como se desee
        for __ in range(max_gen):

            # Ajustar el parámetro alpha
            self.ajuste_alpha(max_gen)

            # Ordernar las fireflies de mayor a menor intensidad
            self.poblacion.sort(key=operator.attrgetter("intensidad"), reverse=True)

            # Mover todas las fireflies
            self.movimiento()

        #  Ordenar la última iteración
        self.poblacion.sort(key=operator.attrgetter("intensidad"), reverse=True)

        # Escoger si se desea obtener el resultado y el valor de la función
        if optim:
            return (self.poblacion[0].pos, self.poblacion[0].brillo)
        else:
            # Sino, regresar la posición más brillante, el resultado
            return self.poblacion[0].pos

    def ajuste_alpha(self, gens):
        """
        Como recomendación del autor original del algortimo, el parámetro alfa manipula
        la aleatoriedad del algoritmo y por tanto es muy útil manejar este valor y actualizarlo
        para acelerar la convergencia del algoritmo.

        Args:
            gens: Valor flotante que afecta al algoritmo en cuestión
        """
        #  Se ajusta el valor de alpha según el autor
        delta = 1.0 - (10.0 ** (-4) / 0.9) ** (1.0 / gens)
        self.alpha_opt *= 1.0 - delta
