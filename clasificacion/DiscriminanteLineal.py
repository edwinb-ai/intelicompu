import numpy as np


class DiscriminanteLineal:
    """Discriminante Lineal del Fisher.

    Esta es una implementación del discriminante lineal de Fisher utilizando
    solamente numpy como dependencia. Este discriminante clasifica un conjunto
    de datos utilizando fronteras de decisión lineales, basado en el teorema de
    Bayes. En particular este discriminante asume que la matriz de covarianza
    es la misma para todos los datos, independiente de su clase.

    Attributes:
        inv_covarianza: La matriz de covarianza inversa de los datos.
        medias: Una lista de las coordenadas de las medias ariteméticas de las clases.
        a_priori: Un arreglo con las probabilidades a priori de cada clase.
        resultado_parcial: Una lista de valores donde se guardan los valores calculados de los
            datos de entrenamiento.
    """

    def __init__(self):
        """Inicializar todos los atributos vacíos siempre."""
        self.inv_covarianza = None
        self.medias = list()
        self.a_priori = np.array([])
        self.resultado_parcial = list()

    def inicializar(self, datos, clases):
        """Se calculan las probabilidades a priori, las medias aritméticas de
        cada clase y la matriz de covarianza inversa de los datos."""
        # Inicializar los valores de las probabilidades a priori
        valores_clases = list(set(clases))
        total_puntos = len(datos)
        for i in valores_clases:
            self.a_priori = np.append(
                self.a_priori, len(clases[clases == i]) / total_puntos
            )
            # Calcular las medias de cada clase
            self.medias.append(datos[clases == i].mean(axis=0))

        # Calcular la matriz de covarianza, es la misma para todos los datos
        covarianza = np.cov(datos, rowvar=False)
        # También calcular la inversa de la matriz de covarianza
        self.inv_covarianza = np.linalg.pinv(covarianza)
        self.medias = np.array(self.medias)

    def entrenamiento(self, datos, clases):
        """El entrenamiento consta de calcular los resultados parciales de sumar
        el producto de las medias con la matriz de covarianza inversa y la suma
        del logaritmo de todas las probabilidades a priori.

        Args:
            datos: El conjunto de datos, un arreglo mutidimensional de numpy.
            clases: Las etiquetas de las clases de datos, un arreglo unidimensional de numpy. 
        """
        # Convertir a np.array
        datos = np.array(datos)
        # Se inicializan todos los valores
        self.inicializar(datos, clases)
        # Se calcula el primer producto interno
        for i in self.medias:
            self.resultado_parcial.append(-0.5 * (i.T @ (self.inv_covarianza @ i)))

        self.resultado_parcial = np.array(self.resultado_parcial)
        # Sumar el logaritmo de las probabilidades a priori
        self.resultado_parcial += np.log(self.a_priori)

    def prediccion(self, datos):
        """Calcula las clases a las que pertenecen los datos dados los valores del
        conjunto de entrenamiento.

        Args:
            datos: El conjunto de datos para clasificar, un arreglo de numpy.

        Returns:
            Un arreglo unidimensional con las clases a las que pertenecen los datos.
            Siempre va a ir desde 0 hasta el número máximo de clase.
        """
        # Convertir a np.array
        datos = np.array(datos)
        # Crear el arreglo vacío con el tamaño correcto
        clasificacion = np.zeros((len(datos), len(self.medias)))
        # Utilizar el resultado entrenado
        for l, i in enumerate(self.medias):
            multiplicacion = self.inv_covarianza @ i
            clasificacion[:, l] = (datos @ multiplicacion) + self.resultado_parcial[l]

        return np.argmax(clasificacion, axis=1)
