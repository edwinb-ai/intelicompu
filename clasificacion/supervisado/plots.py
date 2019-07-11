import matplotlib.pyplot as plt
import numpy as np


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """
    Función para graficar las fronteras de decisión de un clasificador Support Vector
    Machine. El modelo ingresado ya debe de estar entrenado porque se espera que cuente
    con vectores soporte ya determinados para ser graficados.

    Args:
        model: Un objeto tipo model de scikit-learn que sea una instancia de SVC
        ax: Un objeto tipo Plot de matplotlib para generar los ejes de la gráfica
        plot_support: Valor booleano que determina si se muestran los vectores soporte
            en la gráfica
    """
    # Tomar en cuenta si se utilizan los ejes propuestos por el usuario
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Construir la malla para graficar el modelo
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # Graficar la frontera de decisión y los bordes
    ax.contour(
        X, Y, P, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
    )

    # Si se desea, se pueden graficar y mostrar los vectores
    # soporte creados por el modelo para el conjunto de datos
    if plot_support:
        ax.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=100,
            linewidth=1,
            facecolors="none",
            edgecolors="k",
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
