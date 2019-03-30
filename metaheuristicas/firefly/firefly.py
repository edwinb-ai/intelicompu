import numpy as np
import copy
import operator


class Firefly:
    def __init__(self, sup, inf, dim):
        '''
        Esta es la clase base Firefly que crea vectores de posición y valor de una función
        en esta posición. Este constructor recibe los siguientes parámetros:
        sup: límite superior de la firefly
        inf: límite inferior de la firefly
        dim: dimensión del espacio
        El valor de self.intensiddad es el valor de la función en la posición self.posicion.
        '''
        self.intensidad = None
        self.posicion = None
        self.lim_sup = sup
        self.lim_inf = inf
        self.dim_fire = dim

    @property
    def pos(self):
        '''
        Devuelve el valor de la posición.
        '''
        return self.posicion

    @pos.setter
    def pos(self, val):
        '''
        Asigna el valor val al valor de la posición.
        '''
        self.posicion = val

    @property
    def brillo(self):
        '''
        Devuelve el valor de la función (brillo).
        '''
        return self.intensidad

    @brillo.setter
    def brillo(self, val):
        '''
        Asigna el valor val al valor del brillo.
        '''
        self.intensidad = val

    def ini_posicion(self):
        '''
        Genera una posición aleatoria utilizando la dimensión del espacio y los límites
        impuestos desde la construcción de la clase.
        '''
        posicion_aleatoria = (self.lim_sup - self.lim_inf) * np.random.random_sample(self.dim_fire) + self.lim_inf
        self.posicion = posicion_aleatoria


class FAOpt:
    def __init__(self, func, dim, tam_pob, alpha, beta, gamma, inf, sup, args=()):
        '''
        El constructor de esta clase toma los argumentos siguientes:
        func: función objetivo
        dim: dimensión del espacio de soluciones
        tam_pob: tamaño de la población de fireflies
        alpha: parámetro alfa del algoritmo original
        beta: parámetro de brillo inicial
        gamma: parámetro de atracción
        inf: límite inferior de búsqueda
        sup: límite superior de búsqueda
        args: argumentos opcionales para la función objetivo
        '''
        self.alpha_opt = alpha
        self.beta_opt = beta
        self.gamma_opt = gamma
        self.lim_sup = sup
        self.lim_inf = inf
        self.func_opt = func
        self.f_args = args
        self.dim_opt = dim
        self.pob_tam = tam_pob
        # Para crear la población total de posibles soluciones (fireflies) se crea
        # una lista de objetos Firefly con el tamaño especificado por el usuario
        self.poblacion = [Firefly(sup, inf, dim) for i in range(tam_pob)]

    def ini_poblacion(self):
        '''
        Se inicializa la posición y brillo de la población de forma aleatoria.
        '''
        # Por cada firefly en la población
        for i in self.poblacion:
            # Asignarle una posición aleatoria
            i.ini_posicion()

    def movimiento(self):
        '''
        Este método implementa el algoritmo completamente; es único para la clase y NO debe
        de ser llamado afuera de ésta.
        '''
        # Copiar la población anterior
        tmp_poblacion = copy.copy(self.poblacion)

        # Escala del sistema
        escala = abs(self.lim_sup - self.lim_inf)

        # Iterar entre pares de fireflies, utilizando iteradores de Python
        for i in self.poblacion:
            # Se necesita el índice de la población anterior actual para cambiar
            # el valor correspondiente de la población nueva
            for j, k in enumerate(tmp_poblacion):
                # Actualizar la distancia según el algoritmo
                dist = np.power(np.linalg.norm(i.pos - k.pos), 2.0)
                # Si sucede que es una mejor solución
                if i.brillo > k.brillo:
                    # Implementar el Firefly Algorithm
                    beta_new = (1.0 - self.beta_opt) * np.exp(-self.gamma_opt*dist) + self.beta_opt
                    estocastico = self.alpha_opt * (np.random.random_sample(self.dim_opt) - 0.5) * escala
                    tmp_pos = i.pos*(1.0-beta_new) + k.pos*beta_new + estocastico

                    # Verificar que las posiciones no se salgan de los límites
                    np.clip(tmp_pos, self.lim_inf, self.lim_sup, out=tmp_pos)

                    # Actualizar la posición
                    self.poblacion[j].pos = tmp_pos
                    # Actualizar el brillo de esa posición
                    self.poblacion[j].brillo = -self.func_opt(self.poblacion[j].pos, *self.f_args)
        

    def optimizar(self, max_gen, optim=False):
        '''
        Esta es la función que se llama desde afuera y la que implementa todo el algoritmo.
        Parámetros:
        max_gen: número de generaciones o iteraciones para llamar al algoritmo
        optim: controla el hecho si solo devuelve la posición del mínimo, si es verdadero
            devuelve también el valor de la función en esa posición
        '''
        # Inicializar la población, posicion y brillo iniciales
        self.ini_poblacion()
        for i in self.poblacion:
            i.brillo = -self.func_opt(i.pos, *self.f_args)

        # Mover las fireflies tanto como se desee
        for __ in range(max_gen):

            # Ajustar el parámetro alpha
            self.ajuste_alpha(max_gen)

            # Ordernar las fireflies de mayor a menor intensidad
            self.poblacion.sort(key=operator.attrgetter('intensidad'), reverse=True)

            # Mover todas las fireflies
            self.movimiento()
        
        # Ordenar la última iteración
        self.poblacion.sort(key=operator.attrgetter('intensidad'), reverse=True)

        # Escoger si se desea obtener el resultado y el valor de la función
        if optim:
            return (self.poblacion[0].pos, self.poblacion[0].brillo)
        else:
            # Sino, regresar la posición más brillante, el resultado
            return self.poblacion[0].pos


    def ajuste_alpha(self, gens):
        '''
        Como recomendación del autor original del algortimo, el parámetro alfa manipula
        la aleatoriedad del algoritmo y por tanto es muy útil manejar este valor y actualizarlo
        para acelerar la convergencia del algoritmo.
        '''
        # Se ajusta el valor de alpha según el autor
        delta = 1.0 - (10.0**(-4) / 0.9)**(1.0 / gens)
        self.alpha_opt *= (1.0 - delta)
