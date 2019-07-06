from scipy.spatial import KDTree
import numpy as np


class kNearestNeighbors:
    """
    Una implementación del método k-Nearest Neighbors empleando kd-tree
    para la búsqueda de los vecinos más cercanos.

    Attributes:
        kd_tree: Un árbol de búsqueda de vecinos más cercanos, instancia de KDTree
        kn: Un valor entero que determina el número de vecinos más cercanos a buscar.
        dicc_probabilidades: Un diccionario que guarda las probabilidades a priorio de cada clase.
            Las llaves son enteros y los valores son flotantes.
        etiquetas: Un arreglo multidimensional que almacena las clases del conjunto de datos.
    """

    def __init__(self, num_vecinos=3):
        """La inicialización de la clase crea un diccionario vacío para las probabilidades
        a priori de cada clase dentro del conjunto de datos.

        Args:
            num_vecinos: Un valor entero que determinar el número de vecinos más cercanos.
        """
        self.kd_tree = None
        self.kn = num_vecinos
        self.dicc_probabilidades = {}
        self.etiquetas = None

    def entrenamiento(self, datos, clases):
        """Esta función crea el árbol de búsqueda según el conjunto de datos y las clases
        correspondientes. Además inicializa el diccionario de probabilidades y almacena
        una copia de las clases.

        Args:
            datos: Un arreglo multidimensional con el conjunto de datos.
            clases: Un arreglo unidimensional con las etiquetas correspondientes a cada entrada de datos.
        """

        # Inicializar el árbol de búsqueda
        self.kd_tree = KDTree(datos)

        # Inicializar el diccionario de probabilidades
        for i in list(set(clases)):
            self.dicc_probabilidades[i] = 0

        # Guardar las etiquetas
        self.etiquetas = clases

    def predecir(self, datos):
        """Utilizando el árbol de búsqueda creado en el entrenamiento, aquí se realiza la predicción
        emplean self.kn vecinos y utilizando un método de mayoría de votos. Los vecinos se buscan
        dentro del árbol y las probabilidades a priori se calculan para tomar la decisión final.

        Args:
            datos: Un arreglo multidimensional con datos por clasificar.
        """

        resultado = np.zeros(datos.shape[0])

        for n, d in enumerate(datos):
            __, indices = self.kd_tree.query(d, k=self.kn)

            for k in self.dicc_probabilidades.keys():
                for i, j in enumerate(self.etiquetas):
                    for l in indices:
                        # Si el índice es el mismo
                        if i == l:
                            # Y son de la misma clase
                            if j == k:
                                self.dicc_probabilidades[k] += 1

            for i, j in self.dicc_probabilidades.items():
                self.dicc_probabilidades[i] = j / self.kn

            # Siempre se devuelve la mayor probabilidad encontrada entre todas las clases
            resultado[n] = max(
                self.dicc_probabilidades, key=self.dicc_probabilidades.get
            )

            # Reinicar el diccionario de probabilidades
            for i in list(set(self.etiquetas)):
                self.dicc_probabilidades[i] = 0

        return resultado
