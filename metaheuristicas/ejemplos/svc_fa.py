'''
Este archivo implementa un clasificador basado en Máquinas de Soporte Vectorial con
validación cruzada realizada a través de una metaheurística, el Firefly Algorithm.
'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import firefly as fa


def svc_fnc(x, x_tr=None, x_ts=None, y_tr=None, y_ts=None):
    '''
    Esta función es un wrapper del objeto que hace una instancia del clasificador
    basado en SVM. Primero se hace la instancia, luego se ajusta el modelo y por
    último se devuelve un valor de precisión de clasificación.
    Esta función es la que se pretende minimizar, en particular, minimizar el error
    de clasificación.
    Parámetros
    x_tr: Datos de entrenamiento
    y_tr: Etiquetas de los datos de entrenamiento x_tr
    x_ts: Datos de prueba
    y_ts: Etiquetas de los datos de prueba x_ts
    Regresa
    score: Flotante que corresponde al accuracy de clasificación
    '''
    # Crear una instancia del clasificador
    clf = SVC(C=x[0], gamma=x[1])
    # Ajustarlo con los datos de entrenamiento
    clf.fit(x_tr, y_tr)
    # Siempre se buscan valores positivos del accuracy
    score = -1.0*clf.score(x_ts, y_ts)

    return score

# Empezar el código principal
if __name__ == '__main__':
    # Extraer el conjunto de datos, y separarlo en dos
    X, y = load_breast_cancer(return_X_y=True)
    # Se muestra un subconjunto de los datos para visualización
    print('Non-scaled values:', X[:2])

    # Estandarizar los datos entre -1 y 1
    scaler = MinMaxScaler(feature_range=(-1,1))
    X = scaler.fit_transform(X)
    # Nuevamente se muestra un subconjunto de los datos para estandarización
    print('Scaled values:', X[:2])
    print('Target values', y[:2])

    # Es importante separar el conjunto de datos en prueba y entrenamiento
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Separar el conjunto de datos para validación cruzada usando
    # separación estratificada de 10 pliegues
    n_pliegues = 10
    skf = StratifiedKFold(n_splits=n_pliegues)

    # Estos arreglos guardarán los resultados finales
    # Este arreglo guarda los parámetros óptimos, C y gamma
    res_vals = np.zeros(2)
    # Este arreglo guarda el valor de accuracy total
    fnc_total = np.array([])

    # Se comienza a iterar sobre los 10 pliegues de la validación cruzada
    for tr, ts in skf.split(x_train, y_train):
        # Estos son los parámetros de entrada del Firefly Algorithm
        kwargs = {'func': svc_fnc, 'dim': 2, 'tam_pob': 20, 'alpha': 0.9, 'beta': 0.2, 'gamma': 1.0, 
        'inf': 2**(-5), 'sup': 2**5}
        # Se crea una instancia del Firefly Algorithm
        fa_solve = fa.FAOpt(**kwargs, args=(X[tr], X[ts], y[tr], y[ts]))
        # Se llama al método que resuelve la optimización
        res, fnc = fa_solve.optimizar(10, optim=True)
        
        # Se guardan los resultados de cada iteración
        res_vals += res
        fnc_total = np.append(fnc_total, fnc)

    # Los valores de los parámetros C y gamma deben estar normalizados, se divide por
    # el número de pliegues
    res_vals /= n_pliegues
    # Por visualización, se muestran los resultados óptimos (opcional)
    print(res_vals)
    # Y el promedio de accuracy durante la validación cruzada (opcional)
    print(fnc_total.mean())

    # Por último, usando los valores óptimos encontrados se realiza la clasificación final
    clf = SVC(C=res_vals[0], gamma=res_vals[1])
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
    # Y mostrar un reporte de clasificación
    print(classification_report(y_test, clf.predict(x_test)))

