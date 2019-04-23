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

# Regresión (multivariada)

La **_regresión_** es uno de los modelos genéricos básicos del análisis estadístico y por extensión, de la inteligencia computacional. El propósito de la _regresión_ es **predecir** valores.

En el ámbito de la inteligencia computacional la regresión se reduce a un caso de _clasificación_ donde las clases son todos los posibles valores, i.e. las clases son todo el conjunto de los reales.

Por esta razón, existen diversos modelos que atacan diversos tipos de datasets que se pueden analizar. Por ejemplo, cuando se tienen dos variables _linealmente_ correlacionadas el modelo más fácil (y el mejor) sería el modelo estándar lineal que técnicamente minimiza la desviación estándar entre los errores. Pero cuando las variables no son solamente 2, y adicionalmente no están relacionadas linealmente, ¿cuál es el mejor modelo a utilizar?

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
from scipy.stats import randint as sp_randint
from statsmodels.nonparametric.smoothers_lowess import lowess

# Utilizar el tema default de seaborn
sns.set()
```

## Dataset de _diabetes_

Este conjunto de datos proviene de un [estudio](http://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf) de análisis de regresión donde se propone un nuevo (para el año cuando se publicó) método de regresión que intenta mejorar el modelo estándar de regresión por _mínimos cuadrados_. Este método es Lasso-LARS y está incluido en esta libreta.

En general, este conjunto de datos contiene 10 variables, las cuales son:

1. Edad
2. Género
3. Índice de masa corporal
4. Presión arterial promedio
5. $S_1$
6. $S_2$
7. $S_3$
8. $S_4$
9. $S_5$
10. $S_6$

donde las variables $S$ se refieren a mediciones hechas sobre el suero de la sangre. La variable objetivo es el avance de la diabetes en ese paciente en específico después de un año de haber realizado estas mediciones. 

El objetivo de este conjunto de datos es encontrar cuál de todas las variables son las que más contribuyen a la predicción efectiva de la variable objetivo. En particular, en esta libreta se realizará un estudio riguroso de diversos métodos de regresión lineal y se compararán los resultados obtenidos.


## 1. Extracción y exploración básica de datos

```python
# Importar los datos
diab_data = datasets.load_diabetes()
```

```python
# Observar los datos
print(diab_data['data'])
# Y su tamaño
print(diab_data['data'].shape)
```

```python
# Verificar si faltan valores
np.any(np.isnan(diab_data['data']))
```

```python
# Observar cuáles son las categorías
diab_data['feature_names']
```

```python
# Crear una matriz de correlación para explorar los datos y su relación
correlations = np.corrcoef(diab_data['data'])
correlations
```

```python
# Mostrar una matriz de correlación entre los datos
plt.figure(figsize=(12,8))
plt.matshow(correlations)
plt.colorbar()
plt.show()
```

Como un acercamiento básico al análisis de estos datos, realizar una matriz de correlación y graficarla puede ser de ayuda. En este caso se puede observar que muchos de los datos entre sí están altamente correlacionados, definitivamente no se puede extraer un solo par de características para realizar una regresión lineal.

También es útil notar que la matriz de correlación está muy poblada y los colores empiezan a _perderse_ pero es una ayuda visual para la exploración inicial de los datos.

```python
# Separar el conjunto de datos en entrenamiento y prueba, 80/20
x_train, x_test, y_train, y_test = model_selection.train_test_split(diab_data['data'], diab_data['target'],
                                                                   test_size=0.2)
```

## 2. Árboles de decisiones con un _Extremely Randomized Forest_


Los _árboles de decisiones_ son parte de algoritmos que ayudan a encontrar qué características dentro de los datos son los más importantes conocidos como _métodos de **ensambles**_. Esto se logra a través de crear probabilidades según la relación directa entre las características de los datos. Con este tipo de algoritmos se pueden ajustar modelos de regresión para variables que están muy correlacionadas entre sí.

Una de las _ventajas_ de este tipo de algoritmos es que permite realizar regresiones para datos que están correlacionados entre sí, pero su mayor _desventaja_ es que cuando son datos con una dimensionalidad muy alta entonces estos algoritmos tienden a **sobreajustarse.**

```python
# Crear el random forest
forest = ensemble.ExtraTreesRegressor(n_estimators=500)
# Crear un diccionario de posibles valores para realizar la validación cruzada de los 
# hiperparámetros
param_dist = {'max_depth': sp_randint(1, 20),
              'max_features': ['auto', 'sqrt', 'log2'],
             'min_samples_split': sp_randint(2, 10),
              'min_samples_leaf': sp_randint(1, 50)
             }
# Dado que son muchos valores se realizará una validación cruzada aleatoria con 40 muestras
n_iter_search = 40
# ALERTA: Esto puede ralentizar la computadora pues utilizará todos los núcleos disponibles de la computadora
random_search = model_selection.RandomizedSearchCV(forest, param_distributions=param_dist,
                                   scoring='neg_mean_squared_error',n_iter=n_iter_search,
                                    cv=10, iid=False, n_jobs=-1)
# Ajustar el conjunto de datos
random_search.fit(x_train, y_train)
```

```python
# Usar los mejores parámetros y aumentar el número de árboles
best_forest = ensemble.ExtraTreesRegressor(n_estimators=700, **random_search.best_params_)
best_forest.fit(x_train, y_train)
```

```python
# Revisar si hay sobreajuste en el modelo ERF
print(best_forest.score(x_train, y_train))
print(best_forest.score(x_test, y_test))
```

Efectivamente hay _sobreajuste_ como se esperaba dado que el valor de $R^2$ de la predicción de los datos de entrenamiento es superior a los datos de prueba. Sin embargo, no hace falta realizar más ajuste de hiperparámetros, aunque sin duda es un excelente ejercicio, pues no se podrá sacar mucho más provecho de este algoritmo.

Lo que sí se puede obtener es un resumen más detallado de cuáles de estas características son las más importantes para realizar la estimación, que es lo que se hará a continuación.

### 2.1. Ver una gráfica de las características más importantes

Se mencionaba que este tipo de métodos pueden extraer las características más importantes y esto lo realizan a través de un valor conocido como **impureza de Gini**, que para el caso de _regresión_ se escribe como $$I_G(p) = 1 - \sum_{i = 1}^{\infty} p_i$$ donde $p_i$ es la probabilidad de que un valor $i$ sea escogido.

De aquí que las características que tienen un valor más alto de _impuridad de Gini_ son las que se escogen como las que más impacto tienen sobre los datos. Esto se puede ver en una gráfica.

```python
# Extraer las características más importantes
importances = best_forest.feature_importances_
# Calcular el error asociado
std = np.std([tree.feature_importances_ for tree in best_forest.estimators_],
             axis=0)
# Ordenar según el valor más alto de impuridad de Gini
indices = np.argsort(importances)[::-1]

# Muestra una lista de los valores más importantes
print('Ranking de características:')
# Imprimir una lista con la impuridad de Gini de cada característica
for f in range(diab_data['data'].shape[1]):
    print('{0}. feature {1} ({2})'.format(f + 1, indices[f], importances[indices[f]]))

# Graficar la importancia de cada característica como una gráfica de barras,
# en orden descendiente de importancia
plt.figure(figsize=(12, 8))
plt.title('Importancia de características')
plt.bar(range(diab_data['data'].shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(diab_data['data'].shape[1]), indices)
plt.xlim([-1, diab_data['data'].shape[1]])
plt.ylabel('Impuridad de Gini')
# El número de características están en formato de índices de Python
# i.e. 0 es la primera, 1 es la segunda y sucesivamente
plt.xlabel('Características')
plt.show()
```

Algo importante de esta figura es que en realidad dos de las 10 características son las importantes, pero se debe _notar_ lo siguiente: el error asociado es muy grande para cada medición.
Esto se puede deber a que el modelo se está _sobreajustando_ y no tenemos una verdadera medición estadística confiable, que sin duda es importante.

Esta gráfica también dicta que aunque son dos las más importantes, las demás también tienen una cierta contribución a la creación del modelo. Esto se retomará después en esta libreta.

## 2.2 Realizar y visualizar la predicción de _Extremely Randomized Forest_

```python
# Realizar la predicción con este modelo
y_pred = best_forest.predict(x_test)
```

```python
# Crear una visualización de este resultado
plt.figure(figsize=(10,6))
plt.scatter(y_pred, y_test)
# Crear una línea de regresión pesada por los datos para ver el ajuste del modelo
line_plot = lowess(y_pred, y_test)
plt.plot(line_plot[:,0], line_plot[:,1], c='r')
plt.title('Extremely Randomized Forests')
plt.xlabel('Predicción')
plt.ylabel('Datos reales')
plt.show()
```

Aquí se espera ver que todos los datos de predicción y reales formen una línea recta pues esto indicaría que los datos han sido predichos correctamente. Sin embargo se puede ver que es muy difícil de tener este resultado con este modelo _sobreajustado_.

## 3. Regresión mediante SVM (Máquinas de soporte vectorial)

Uno de los métodos de inteligencia computacional más prominentes son las SVMs que permiten trabajar con datos de alta dimensionalidad. Aún mejor, este algoritmo se puede extender a regresión y es lo que se realiza aquí. Este método es un fuerte candidato para realizar una buena regresión, pero este método no creará un separación de características y una desventaja de este algoritmo es que el ajuste de hiperparámetros puede ser muy costoso computacionalmente.

```python
# Crear diccionario de parámetros
params = {'C': [10, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
         'epsilon': [0.005, 0.01, 0.05, 0.1, 0.2, 0.3]}
# Crear la SVM para regresión con kernel radial
sv_reg = svm.SVR(kernel='rbf')
# Realizar validación cruzada para encontrar el mejor regresor
best_svr = model_selection.GridSearchCV(sv_reg, param_grid=params, 
                                        scoring='neg_mean_squared_error',iid=False, cv=10, n_jobs=-1)
# Entrenar el modelo
best_svr.fit(x_train, y_train)
```

```python
# Encontrar los valores predichos por el modelo
y_pred = best_svr.predict(x_test)
```

```python
# Revisar si hay algún problema con el ajuste de SVM
print(best_svr.best_estimator_.score(x_train, y_train))
print(best_svr.best_estimator_.score(x_test, y_test))
```

Según este resultado anterior, existe un _sobreajuste_, aunque ligero, con el modelo. Sin embargo, este modelo ya está realizando un predicción aceptable y esto se puede verificar con la gráfica siguiente.

```python
# Crear una visualización de este resultado
plt.figure(figsize=(10,6))
plt.scatter(y_pred, y_test)
line_plot = lowess(y_pred, y_test)
plt.plot(line_plot[:,0], line_plot[:,1], c='r')
plt.title('Regresión con SVM, rbf kernel')
plt.xlabel('Predicción')
plt.ylabel('Datos reales')
plt.show()
```

Se esperaba tener un línea recta nuevamente, pero no existe esta relación lineal entre los datos predichos y los reales.

## 4. Gradient Boosting

El método de _ensamble_ de **Gradient Boosting** es la mejor substancial de los _Random Forests_. Básicamente este tipo de métodos lo que hacen es construir muchos árboles de decisión, pero los errores estadísticos (según una métrica estadística arbitraria) son minimizados paso por paso en cada una de las etapas de construcción de los árboles. La _ventaja_ de este método es que puede discernir entre diferentes características relacionadas entre sí, pero su mayor _desventaja_ es que es lento y es complicado de ajustar los hiperparámetros para evitar el _sobreajuste_.

```python
# Parámetros para buscar
params = {'loss': ['ls', 'huber', 'lad'],
         'learning_rate': [0.1, 0.05, 0.02, 0.01],
         'max_depth': sp_randint(1,10),
          'min_samples_leaf': sp_randint(2,100),
          'max_features': sp_randint(2,11)
         }
# Crear el modelo
gr_est = ensemble.GradientBoostingRegressor(n_estimators=1000, subsample=0.5)
# Dado que son muchos valores a buscar, se realiza la búsqueda aleatoria
n_search = 40
gr_cv = model_selection.RandomizedSearchCV(gr_est, param_distributions=params, n_iter=n_search, cv=10, iid=False,
                                     scoring='neg_mean_squared_error', n_jobs=-1)
gr_cv.fit(x_train, y_train)
```

```python
# ¿Cuáles son los mejores parámetros encontrados?
gr_cv.best_params_
```

## 4.1. Visualización de sobreajuste del modelo

```python
# Crear el modelo según los mejores parámetros encontrados
gr_est = ensemble.GradientBoostingRegressor(n_estimators=1000, subsample=0.5, **gr_cv.best_params_)
gr_est.fit(x_train, y_train)
# Se crea un arreglo vacío para guardar los valores de predicción
test_score = np.empty_like(gr_est.estimators_)
# Se itera sobre cada valor predicho y su valor de la métrica escogida, que en este caso es 'ls'
for i, pred in enumerate(gr_est.staged_predict(x_test)):
    test_score[i] = gr_est.loss_(y_test, pred)
# Se grafica la figura para prueba y entrenamiento
plt.figure(figsize=(10,6))
plt.plot(np.arange(1000)+1, test_score, label='Prueba')
plt.plot(np.arange(1000)+1, gr_est.train_score_, label='Entrenamiento')
plt.xlabel('Número de árboles (n_estimators)')
plt.ylabel('Erro de predicción (Loss cost)')
plt.legend(loc='best')
```

Hay sobrejuste del modelo, pero dado que es computacionalmente costoso ajustar este modelo se dejará así en esta libreta, aunque no es el modelo ideal y falta trabajo por realizar para que este modelo tenga un buen desempeño.

```python
# Realizar la predicción con GBR
y_pred = gr_est.predict(x_test)

# Crear una visualización de este resultado
plt.figure(figsize=(10,6))
plt.scatter(y_pred, y_test)
line_plot = lowess(y_pred, y_test)
plt.plot(line_plot[:,0], line_plot[:,1], c='r')
plt.title('Gradient Boosting')
plt.xlabel('Predicción')
plt.ylabel('Datos reales')
plt.show()
```

Nuevamente, no se encuentra una relación lineal entre los datos predichos y reales.

## 5. Lasso-LARS

El algoritmo de regresión **LASSO** (Least Absolute Shrinkage and Selection Operator) es un método de regresión muy robusto que se ajusta a datos multidimensionales altamente correlacionados entre sí.

Matemáticamente, este modelo es un problema de optimización que intenta minimizar una función de costo con un término adicional de penalización. Para mayor información, leer el [artículo original.](http://statweb.stanford.edu/~tibs/lasso/lasso.pdf) Normalmente este algoritmo se minimiza utilizando el método numérico de [_coordinate descent_](https://en.wikipedia.org/wiki/Coordinate_descent) dado que la función de costo contiene un valor absoluto que ocasionaría problemas para el método tradicional de _descenso de gradiente._

En particular, en esta libreta se utiliza un algoritmo diferente para realizar la minimización de Lasso, llamado **LARS** (Least Angle Regression), que fue creado con este conjunto de datos en mente, por lo que se predice que este sea el mejor método para este trabajo. El [artículo](https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf) original contiene el detalle matemático de cómo funciona este método.

```python
# Utilizar la versión con validación cruzada integrada
reg = linear_model.LassoLarsCV(fit_intercept=True, max_iter=500, max_n_alphas=1000, cv=10, n_jobs=-1)
reg.fit(x_train, y_train)
```

```python
print(reg.score(x_train, y_train))
print(reg.score(x_test, y_test))
```

```python
# Calcular el valor del error cuadrático medio de Lasso-LARS
lasso_mse = metrics.mean_squared_error(y_test, reg.predict(x_test))
lasso_mse
```

```python
# Realizar la predicción con Lasso-LARS
y_pred = reg.predict(x_test)

# Crear una visualización de este resultado
plt.figure(figsize=(10,6))
plt.scatter(y_pred, y_test)
line_plot = lowess(y_pred, y_test)
plt.plot(line_plot[:,0], line_plot[:,1], c='r')
plt.title('Lasso-LARS')
plt.xlabel('Predicción')
plt.ylabel('Datos reales')
plt.show()
```

```python
print("Creando el camino de regularización para Lasso ...")
__, active, coefs = linear_model.lars_path(diab_data['data'], diab_data['target'], method='lasso', verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.figure(figsize=(10,6))
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coeficientes')
plt.title('Camino de LASSO')
plt.axis('tight')
plt.show()
```

Una de las grandes ventajas de la regresión Lasso es que puede discernir entre todas las características del conjunto de datos y detallar cuáles son más importantes. La gráfica anterior muestra este resultado.

En particular, cada línea dentro de la gráfica muestra el peso de cada característica que contiene el conjunto de dato y los pesos que tienen para atribuir al modelo de regresión que realizará la predicción.

Se pueden notar las siguientes cosas:

1. **Dos** características empiezan por tener todo el peso en las primeras iteraciones del modelo Lasso. Esto se había visto anteriormente con los _Extremely Randomized Forests_.
2. Cuando el modelo empieza a tomar en cuenta las demás características, empiezan a tener pesos semejantes, aunque menores.
3. Al final, el modelo utiliza __9__ de las **10** características totales para realiza la predicción pertinente.

Con esto se puede _concluir_ lo siguiente:

__9__ de las **10** características son necesarias para realizar una predicción adecuada del modelo, pero aún así, cualquier modelo realizará una predicción pobre por debajo del 60% de las veces. Por otro lado, solamente **2** características tienen el peso fundamental de la relación entre la predicción de las variables como tal.

```python
print('ExtraRandomForest RMSE: {0}'.format(np.sqrt(np.abs(random_search.best_score_))))
print('SVR RMSE: {0}'.format(np.sqrt(np.abs(best_svr.best_score_))))
print('Gradient Boosting RMSE: {0}'.format(np.sqrt(np.abs(gr_cv.best_score_))))
print('Lasso-LARS RMSE: {0}'.format(np.sqrt(np.abs(lasso_mse))))
```

El modelo que tiene el menor error es el que mejor puede predecir, dentro de un intervalo o margen de error. En este caso, el mejor es **Lasso-LARS**, como se esperaba. Las SVMs también tuvieron un buen desempeño, pero hizo falta la mejora de hiperparámetros, pues es muy _difícil_ que las SVMs sean mejores que los métodos de _Gradient Boosting_ en conjunto de datos donde existe una alta correlación. Cabe mencionar que hizo falta ajusta los hiperparámetros adicionales del _Gradient Boosting_.
