import numpy as np


class GNaiveBayes:
    """Clasificador Gaussian Naive Bayes.

    Esta es una implementación del clasificador Gaussian Naive Bayes, basado en el teorma de Bayes
    y asumiendo que los datos siguen una distribución normal Gaussiana. Para emplear este clase primero
    se debe instanciar, entrenar con un conjunto de datos y luego predecir las clases de un conjunto
    de datos diferente.

    Attributes:
        lista_inv_covarianza: Una lista de arreglos numpy que contiene las matrices de covarianza inversas
            de cada clase en el conjunto de datos.
        lista_covarianza: Una lista de arreglos numpy que contiene las matrices de covarianza de cada clase.
        medias: Una lista de arreglos de numpy que contiene la media aritmética de cada clase.
        a_prior: Un arreglo numpy que contiene las probabilidades a priori de cada clase, esto es, el número
            total de datos de la clase dividido por el número total de datos.
        resultado_parcial: Una lista de arreglos numpy que almacena los cálculos que se generan conforme
            se implementa el método.
    """

    def __init__(self):
        """Inicializa el objeto GNaiveBayes creando listas vacías y arreglos numpy
        vacíos según sea el caso.
        """
        self.lista_inv_covarianza = []
        self.lista_covarianza = []
        self.medias = []
        self.a_priori = np.array([])
        self.resultado_parcial = []

    def inicializar(self, datos, clases):
        """Calcula las medias de cada clase, las matrices de covarianza y sus inversas y todo
        lo almacena en los atributos correspondientes dentro de la instancia.

        Args:
            datos: Un arreglo numpy con los datos y sus características.
            clases: Un arreglo numpy con las etiquetas de las clases a las que pertenecen
                en el arreglo datos.
        """
        # Inicializar los valores de las probabilidades a priori
        valores_clases = list(set(clases))
        total_puntos = len(datos)
        for i in valores_clases:
            self.a_priori = np.append(
                self.a_priori, len(clases[clases == i]) / total_puntos
            )
            # Calcular las medias de cada clase
            self.medias.append(datos[clases == i].mean(axis=0))
            # Calcular la matriz de covarianza, diferente para cada clase
            covarianza = np.cov(datos[clases == i], rowvar=False)
            self.lista_covarianza.append(covarianza)
            # También calcular la inversa de la matriz de covarianza
            self.lista_inv_covarianza.append(np.linalg.pinv(covarianza))
        # Convertir a ndarray para aprovechar numpy
        self.medias = np.array(self.medias)

    def entrenamiento(self, datos, clases):
        """Se calculan los primeros valores del método del clasificador utilizando
        un conjunto de datos de entrenamiento. Aquí se calcula el pseudo determinante
        de cada matriz de covarianza por cada clase, así como sumar las probabilidades
        a priori.

        Args:
            datos: Un arreglo numpy con los datos de entrenamiento y sus características.
            clases: Un arreglo numpy con las etiquetas de cada clase presente en el arreglo datos.
        """
        # Convertir a np.array
        datos = np.array(datos)
        # Se inicializan todos los valores
        self.inicializar(datos, clases)
        # Se calcula el primer producto interno
        for i in self.lista_covarianza:
            # Calcular el pseudo-determinante mediante el producto de 
            # eigenvalores positivos
            eig_values = np.linalg.eigvals(i)
            pseudo_determinante = np.product(eig_values[eig_values > 0])
            self.resultado_parcial.append(-0.5 * np.log(pseudo_determinante))
        # Convertir a arreglo de numpy
        self.resultado_parcial = np.array(self.resultado_parcial)
        # Sumar el logaritmo de las probabilidades a priori y un término adicional
        # de la distribución normal
        self.resultado_parcial += np.log(self.a_priori) - 0.5 * datos.shape[0] * np.log(
            2.0 * np.pi
        )

    def prediccion(self, datos):
        """Utilizando los valores de entrenamiento se calcula la distancia de Mahalanobis
        de cada punto a predecir y se devuelve el valor con la mayor probabilidad a posteriori
        como la clase a la que pertenece ese valor.

        Args:
            datos: Un arreglo numpy con los datos y características a predecir.

        Returns:
            Un arreglo numpy con las clases a la que pertenece cada valor del arreglo datos.
        """
        # Convertir a ndarray para aprovechar las características de numpy
        datos = np.array(datos)
        # Crear el arreglo vacío con el tamaño correcto
        clasificacion = np.zeros((len(datos), len(self.medias)))
        # Utilizar el resultado entrenado
        for l, i in enumerate(self.medias):
            for k, d in enumerate(datos):
                # Todo esto corresponde a la distancia de Mahalanobis
                delta_datos = d - i
                multiplicacion = self.lista_inv_covarianza[l] @ delta_datos
                clasificacion[k, l] = (
                    -0.5 * (delta_datos.T @ multiplicacion) + self.resultado_parcial[l]
                )

        return np.argmax(clasificacion, axis=1)
