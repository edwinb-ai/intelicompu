import numpy as np
from scipy.spatial.distance import sqeuclidean
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class kMeans:
    """
    Implementación del algoritmo k-Means para clasificación no supervisada. Empleando las medias
    aritméticas y la distancia euclidiana se categorizan los datos según el número de clases
    que se deseen. Este implementación tiene como criterio de paro el número de iteraciones, pero se
    puede modificar para aceptar también una tolerancia de error.

    Attributes:
        clusters: Diccionario que guarda las clases y los puntos que pertenecen a cada una.
        centroides: Arreglo numpy que tiene tantos centroides como clases. De tamaño (k, d)
            donde k son las clases y d el número de características de los datos.
        num_clases: Valor entero que corresponde al número de clases a clasificar.
        iteraciones: Valor entero para el número máximo de iteraciones.
    """

    def __init__(self, k_comps=3, max_iter=100):
        """Cuando se crea una instancia se incializan los atributos empleando los
        valores vacíos de cada tipo.

        Args:
            k_comps: Valor entero para asignar el número de clases.
            max_iter: Valor entero para determinar el número máximo de iteraciones.
        """
        self.clusters = {}
        self.centroides = None
        self.num_clases = k_comps
        self.iteraciones = max_iter

    def inicializar(self, datos):
        """Cuando se inicializa, se obtienen los primeros centros de forma
        aleatoria y se guardan.

        Args:
            datos: Arreglo numpy con todos los datos a clasificar.
        """

        # Guardar la segunda dimensión porque corresponde al
        # número de características
        dim_datos = datos.shape[1]
        # Crear los arreglos vacíos por clase
        k_centroide = np.zeros((self.num_clases, dim_datos))
        for k in range(self.num_clases):
            # Inicializar aleatoriamente los centroides
            for d in range(dim_datos):
                centro = np.random.uniform(np.min(datos[:, d]), np.max(datos[:, d]))
                k_centroide[k, d] = centro
        # Guardar los centros
        self.centroides = k_centroide

    def clasificar(self, datos):
        """Para clasificar se emplea el criterio de k-means, iterar hasta llegar
        al número máximo de iteraciones, calculando el centro de cada clase y determinar
        los nuevos subconjuntos como clases.

        Args:
            datos: Arreglo numpy con todos los datos a clasificar.

        Returns:
            y_etiquetas: Arreglo numpy con los valores de las etiquetas de cada entrada
                de datos perteneciente a una clase particular.
        """

        # Crear centros iniciales
        self.inicializar(datos)
        # Crear un arreglo vacío de distancias para guardarlas en el ciclo
        distancia = np.zeros(self.num_clases)

        for _ in range(self.iteraciones):
            # Reinicializar el diccionario de clases
            for k in range(self.num_clases):
                self.clusters[k] = []

            # Calcular distancias
            for fila in datos:
                for k in range(self.num_clases):
                    distancia[k] = sqeuclidean(fila, self.centroides[k, :])
                idx_dminima = np.argmin(distancia)
                self.clusters[idx_dminima].append(fila)

            # Calcular los nuevos centros
            for k, v in self.clusters.items():
                self.clusters[k] = np.array(v)
                self.centroides[k] = np.mean(v, axis=0)

        # Crear arreglo de etiquetas
        y_etiquetas = np.zeros(datos.shape[0])
        for k, v in self.clusters.items():
            for dato in v:
                idx_dim, _ = np.where(datos == dato)
                y_etiquetas[idx_dim[1]] = int(k)

        return y_etiquetas
