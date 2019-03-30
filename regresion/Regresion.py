#%% [markdown]
# # Regresión (Simple)

#%% [markdown]
# ## 1. Introducción
# La _regresión lineal_ es un concepto familiar de la estadística paramétrica donde se aplican muchos de los principios y teoremas
# fundamentales de esta área. Existen diversos propósitos para realizar una regresión, pero en particular se pueden reducir a dos
# principales:
#
# 1. _Ajustar_ la mejor linea recta a un conjunto de datos, y con esto se cumplen dos subobjetivos:
# 
# a. Encontrar los _parámetros_ de la mejor linea recta. Estos pueden ser lo que se busca originalmente en algún tipo de experimento diseñado.
# b. Realizar predicciones en base a los datos y la mejor linea recta. Esta linea recta se puede generalizar y dado un cierto _intervalo de
# confianza esto puede significar que las predicciones son válidas.
#
# 2. Encontrar una _correlación_ entre las variables que se encuentran en el conjunto de datos. En diversos caso de problemas reales sucede
# que se tienen mediciones y _no_ se conoce cuáles de estos datos tienen una _correlación lineal_ por lo que realizar la regresión puede
# ayudar a separar estas variables entre sí.
#
#
# ### Modelo matemático
# Sea $\mathbf{X} \in \mathbb{R}$ el conjunto de datos de la variable _independiente_, y sea $\mathbf{Y} \in \mathbb{R}$ el conjunto de datos de la
# variable _dependiente_, entonces el _modelo lineal_ que se busca obtener es el siguiente: $$ \hat{Y} = \alpha + \beta X ,$$ donde $\hat{Y}$ representa
# el valor de la variable _dependiente_ **obtenida** por el modelo lineal. Nótese que **no** es la misma variable original, pues es posible que _no_
# sean los mismos valores.
# Una vez descrito el modelo, quedan por determinar los dos parámetros $\alpha$ y $\beta,$ ¿cómo se realiza esto?
#
# ### Minimización de la _función de costo_
# Normalmente se necesitan dos ecuaciones diferentes para dos incógnitas, y esto es lo que se planea realizar. Al menos es una de las formas de realizar
# esta determinación. Pero este modelo de _regresión lineal ordinaria_ se basa en que los **errores** deben de ser minimizados. ¿Qué tipo de errrores?
# Es esta pregunta la que introduce la siguiente _función de costo_ que debe de ser minimizada: 
# $$ RSS = \sum_{i=1}^{N} \hat{\epsilon_i}^2 =  \sum_{i=1}^{N} \left(y_i - \left(\alpha + \beta x_i \right)\right)^2 ,$$
# donde $N$ es el número total de valores tanto en $\mathbf{X}$ como en $\mathbf{Y}$ y $\hat{\epsilon_i}$ son los errores de _predicción_ respecto a los
# _reales_, esto es, entre $\mathbf{Y}$ y $\mathbf{\hat{Y}}$.
# Existe una forma cerrada (expresiones analíticas que se obtienen a partir
# de teoremas del cálculo diferencial), sin embargo esto puede ser minimizado también mediante métodos numéricos (como [descenso de gradiente](https://en.wikipedia.org/wiki/Gradient_descent)
# o el [método de Newton](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization), entre muchos otros). En particular, esta libreta _no_ tiene como
# objetivo realizar y demostrar este tipo de análisis. Para mayor información, visitar las referencias 1 y 2.
#
# ### Las diferentes hipótesis de la regresión lineal ordinaria
# Para que este modelo sea viable, confiable y aplicable, se deben cumplir varias hipótesis las cuales son difíciles de satisfacer en situaciones de datos reales
# pero que es común encontrar que se satisfacen en experimentos diseñados y/o controlados. Las hipótesis son:
#
# 1. El **valor esperado** de los errores $\epsilon$ debe cumplir que $E\left(\epsilon \vert x\right) = 0.$
# 2. La **varianza** de los errores debe de ser _finita_, _constante_ y _conocida_ tal que cumpla $V\left(\epsilon \vert x\right) = \sigma^2 < \infty.$
# 3. **No** existe una _correlación_ entre los errores tal que $Cov\left(\epsilon_i \vert \epsilon_j\right) = 0.$
#
#
# Lo más _común_ es asumir que los _errores_ siguen una distribución _normal_ tal que $\epsilon \vert x \sim \mathcal{N}\left(0, \sigma^2\right)$
# y esta es la hipótesis más _fuerte_ y _restrictiva_ para este modelo; cuando esto no sucede se deben emplear métodos _no paramétricos_ que puedan resolver
# esta limitante, o bien, acudir a métodos más robustos que pertenecen a la rama de la inteligencia computacional como se verá posteriormente en esta libreta.
#%% [markdown]
# ## 2. _Support Vector Regression_
# NOTA: **NO** se cubrirá toda la teoría de las máquinas de soporte vectorial (SVM); para más información, revisar la referencia 3.
#
# ### Motivación
# Las SVM han probado ser un algoritmo robusto, fantástico y muy adecuado para problemas de _clasificación_. Sin embargo, V. Vapnik creó un algoritmo adicional
# donde atacaba directamente el problema de la _regresión_ mediante un problema de _clasificación_, efectivamente aplicando las SVM a este problema y creando
# un algoritmo extremadamente robusto para crear regresiones lineales y no lineales sin las restricciones que contiene el modelo de regresión lineal ordinario.
#
# ### Modelo matemático
# NOTA: **No** se llevarán a cabo las demostraciones en esta libreta, y la presentación de las ecuaciones será meramente ilustrativa.
# Para aquella persona que desee conocer más al respecto, leer con detalle las referencias 4 y 5.
# 
# La formulación estándar de las SVM para regresión se conoce como $\epsilon-SVR$ (epsilon support vector regression, en inglés). Para realizar la formulación
# se debe tener un conjunto de datos ${(x_1, y_1),\cdots,(x_n,y_n)} \subset \mathbb{H}\times\mathbb{R}$, donde los valores $x_1 \cdots x_n$ son elementos
# de un espacio general $\mathbb{H}$ dado que en el caso más general pueden contener cualquier número de _características_, y no están restringidas
# a pertenecer el conjunto $\mathbb{R}$.
# Es muy importante notar que a diferencia del caso de clasificación, aquí las _clases_ (los valores $y$) son todos los números reales.
# Ahora bien, igual que en el caso de la regresión lineal ordinaria se busca encontrar una relación lineal tal que 
# $$ f(x) = \langle \omega, x\rangle + b$$, con $\omega \in \mathbb{H}$ y $x \in \mathbb{R}.$
# Aquí, $\langle \cdot \rangle$ representa el [_producto interno_](http://mathworld.wolfram.com/InnerProduct.html) en el espacio $\mathbb{H}$.
# Al igual que en el caso de la _clasificación_, se requiere de un problema de _minimización_, esto es, el problema de _regresión_ se reduce a un problema
# de _clasificación_ que a su vez se reduce a un problema de _minimización._
#
# ### El problema de minimización
# NOTA: **No** se llevarán a cabo las demostraciones en esta libreta, y la presentación de las ecuaciones será meramente ilustrativa.
# Para aquella persona que desee conocer más al respecto, leer con detalle la referencia 3 y 6.
#
# Siguiendo la discusión anterior, el problema a minimizar es el siguiente:
# $$
# \begin{aligned}
# & \text{minimizar} \quad
# \frac{1}{2} \vert \vert \omega \vert \vert^2 \\[2ex]
# & \text{sujeto a }
# \begin{cases}
# y_i - \langle \omega, x_i \rangle - b \leq \epsilon, \\[2ex]
# \langle \omega, x_i \rangle + b - y_i \leq \epsilon .
# \end{cases}
# \end{aligned}
# $$
# 
# Este problema es en realidad un problema de _optimización convexa_ que es _viable_ cuando $f$ existe y aproxima todo los pares $(x_n,y_n)$ con precisión
# $\epsilon.$ Sin embargo, como en el caso de SVM de margen suave, existe una versión análoga para el caso de SVR donde se introducen dos variables
# que permitan un cierto margen de error, por lo que se puede reformular todo el problema de optimización anterior por el siguiente:
# $$
# \begin{aligned}
# & \text{minimizar} \quad
# \frac{1}{2} \vert \vert \omega \vert \vert^2 + C \sum_{i=1}^{N} (\xi_i - \xi_i^{\star}) \\[2ex]
# & \text{sujeto a }
# \begin{cases}
# y_i - \langle \omega, x_i \rangle - b &\leq \epsilon + \xi_i, \\[2ex]
# \langle \omega, x_i \rangle + b - y_i &\leq \epsilon + \xi_i^{\star}, \\[2ex]
# \xi_i, \xi_i^{\star} &\geq 0 .
# \end{cases}
# \end{aligned}
# $$
#
# Al final, y utilizando el mismo método de multiplicadores de Lagrange (ver la referencia 3), se obtiene la función con la que se encuentra el **modelo
# lineal**:
# $$ f(x) = \sum_{i=1}^{N} (\alpha_i - \alpha_i^{\star}) \langle x_i, x \rangle + b ,$$
# donde $\alpha_i$ y $\alpha_i^{\star}$ son los _multiplicadores de Lagrange._ 
# Para el caso **no lineal** sólo basta modificar la ecuación anterior para incluir una _función kernel_ que realice un mapeo del espacio $\mathbb{H}$
# tal que se obtiene la siguiente ecuación:
# $$ f(x) = \sum_{i=1}^{N} (\alpha_i - \alpha_i^{\star}) k(x_i, x) + b , $$ donde se tiene que
# $$\omega = \sum_{i=1}^{N} (\alpha_i - \alpha_i^{\star}) \phi(x_i)$$
# y $k(x_i, x) = \vphi(x_i) \vphi(x)$ es un _kernel de Mercer_ (revisar la referencia 4 para más información).
# ¿Cómo se determinan? Claro está que sin estos valores **no** se puede
# encontrar el modelo lineal buscado.
#
# ### Función de riesgo
# NOTA: **No** se llevarán a cabo las demostraciones en esta libreta, y la presentación de las ecuaciones será meramente ilustrativa.
# Para aquella persona que desee conocer más al respecto, leer con detalle la referencia 3 y 6.
#
# 
#%%
# Realizar todas las importaciones necesarias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_int

# Definir el conjunto de colores de Seaborn
sns.set()
#%%
# Cargar el Firefly Algorithm, la metaheurística para encontrar los hiperparámetros adecuados
# NOTA: cambiar este directorio por el adecuado si es que esta libreta se corre de nuevo
%load '/home/edwin/Documents/UG-Fisica/Octavo_Semestre/Desarrollo_Experimental/ejercicios/firefly/firefly.py'

import firefly.firefly as fa
#%% [markdown]
# ## 1. Complejidad del modelo (sobreajuste)
# En esta primera sección se implementa una regresión utilizando _máquinas de soporte vectorial_ (MSV) aplicadas para el problema de
# regresión 
#%%
# Crear un problema simple de regresión, se espera que sea totalemente
# correlacionado y lineal
X, y = make_regression(n_samples=400, n_features=1, n_informative=1, bias=5.0, noise=14.0, random_state=15)
#%%
# Graficar estos datos para ver con qué se está trabajando
plt.figure(figsize=(10, 8))
plt.scatter(X, y)
plt.show()
#%% [markdown]
# Como se esperaba el modelo es lineal... (hacer más énfasis de esto, y el ruido)
#%%
# Separar el conjunto de datos en prueba y entrenamiento
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
#%% [markdown]
# ## 1.1. Kernel _lineal_
# Mencionar alguna hipótesis de que este kernel es el que se espera que sea mejor
#%%
# Crear un regresor
lineal = SVR(kernel='linear')
# Definir parámetros para la validación cruzada
params = {'C': sp_int(2**(-15), 2**15), 'epsilon': sp_int(2**(-8), 2**8)}
# Realizar la validación cruzada aleatoria
reg_lin = RandomizedSearchCV(lineal, param_distributions=params, scoring='neg_mean_squared_error',
n_iter=25, cv=10, iid=False, n_jobs=-1)
# Ajustar el modelo
reg_lin.fit(x_train, y_train)
#%% [markdown]
# Una nota sobre RandomizedSearchCV (porqué usarlo, porqué sirve, y cuando es útil usarlo)
#%%
# Mostrar los méjores parámetros
lineal = reg_lin.best_estimator_
print(lineal)

# Mostrar el R^2
print(lineal.score(x_train, y_train))
#%% [markdown]
# Mencionar que $R^2$ no es la mejor métrica y que se prefiere el MSE
#%%
# Con este regresor, realizar predicciones
y_pred = lineal.predict(x_test)
# Mostrar solo algunos valores para visualización
y_pred[:5]
#%%
# Calcular el error entre entrenamiento y prueba
print('MSE para entrenamiento: {0}'.format(np.abs(mean_squared_error(y_train, lineal.predict(x_train)))))
print('MSE para prueba: {0}'.format(np.abs(mean_squared_error(y_test, y_pred))))
#%% [markdown]
# Dado que los errores son muy semejantes, **NO** existe sobreajuste del modelo. Elaborar este argumento un poco más...
#%% [markdown]
# ## 1.2. Kernel _base radial._
# Mencionar lo que se espera, que no va a ser muy bueno porque no es el espacio correcto
# (tener un mejor argumento, con referencia si es posible)
#%% [markdown]
# ## Sección _bonus_. Hablar sobre metaheurísticas (muy rápido) y el método que se pretende seguir, mencionar porqué es útil
# porqué será necesario utilizarlo...
#%% [markdown]
# Se envuelve al objeto SVR en un función para después hacer uso de una metaheurística
# y poder realizar ajuste adecuado de hiperparámetros.
#%%
def svr_fnc(x, x_tr=None, x_ts=None, y_tr=None, y_ts=None):
    # Crear una instancia del clasificador
    reg = SVR(kernel='rbf', gamma=x[0], C=x[1], epsilon=x[2])
    # Ajustarlo con los datos de entrenamiento
    reg.fit(x_tr, y_tr)
    y_pred = reg.predict(x_ts)
    # Siempre se buscan valores positivos del accuracy
    score = np.sqrt(np.abs(mean_squared_error(y_ts, y_pred)))

    return score
#%%
# Separar el conjunto de datos para validación cruzada usando
# separación de 10 pliegues
n_pliegues = 10
skf = KFold(n_splits=n_pliegues)

# Estos arreglos guardarán los resultados finales
# Este arreglo guarda los parámetros óptimos, C y gamma
res_vals = np.zeros(3)
# Este arreglo guarda el valor de accuracy total
fnc_total = np.array([])

# Se comienza a iterar sobre los 10 pliegues de la validación cruzada
for tr, ts in skf.split(x_train, y_train):
    # Estos son los parámetros de entrada del Firefly Algorithm
    kwargs = {'func': svr_fnc, 'dim': 3, 'tam_pob': 20, 'alpha': 0.9, 'beta': 0.2, 'gamma': 1.0, 
    'inf': 2**(-4), 'sup': 2**3}
    # Se crea una instancia del Firefly Algorithm
    fa_solve = fa.FAOpt(**kwargs, args=(X[tr], X[ts], y[tr], y[ts]))
    # Se llama al método que resuelve la optimización
    res = fa_solve.optimizar(10)
    
    # Se guardan los resultados de cada iteración
    res_vals += res
    fnc_total = np.append(fnc_total)

# Los valores de los parámetros C y gamma deben estar normalizados, se divide por
# el número de pliegues
res_vals /= n_pliegues
# Por visualización, se muestran los resultados óptimos (opcional)
print(res_vals)
#%%
# Por último, usando los valores óptimos encontrados se realiza la clasificación final
reg_rbf = SVR(kernel='rbf', gamma=res_vals[0], C=res_vals[1], epsilon=res_vals[2])
reg_rbf.fit(x_train, y_train)
# Con este regresor, realizar predicciones
y_pred = reg_rbf.predict(x_test)
print(reg_rbf.score(x_test, y_test))
#%%
# Calcular el error entre entrenamiento y prueba
print('RMSE para entrenamiento: {0}'.format(np.sqrt(np.abs(mean_squared_error(y_train, reg_rbf.predict(x_train))))))
print('RMSE para prueba: {0}'.format(np.sqrt(np.abs(mean_squared_error(y_test, y_pred)))))
#%% [markdown]
# Mencionar la variación muy grande entre errores para fortalecer el argumento de sobreajuste...
#%% [markdown]
# Ahora se puede visualizar el resultado graficando ambos regresores
#%%
# Código tomado de
# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py
# y adaptado para esta libreta
lw = 2

svrs = [lineal, reg_rbf]
kernel_label = ['Lineal', 'RBF']
model_color = ['m', 'c']

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), sharey=True)
for ix, svr in enumerate(svrs):
    # axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw,
    #               label='{} model'.format(kernel_label[ix]))
    axes[ix].scatter(X, svr.fit(X, y).predict(X), color='r', lw=lw,
                  label='{} model'.format(kernel_label[ix]))
    axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                     edgecolor=model_color[ix], s=50,
                     label='{} support vectors'.format(kernel_label[ix]))
    axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                     y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                     facecolor="none", edgecolor="k", s=50,
                     label='other training data')
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)

fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()
#%% [markdown]
# ## Conclusiones
# 1. Hablar acerca del sobreajuste de los modelos...
# 2. Hablar del número de vectores soporte para cada kernel
# 3. Hablar de la predicción de cada uno
# 4. Hablar de la complejidad (número de hiperparámetros de cada modelo)
#%% [markdown]
# ## 2. Aplicación: aceleración de la gravedad (_revisited_)
# Mencionar sobre este experimento, poner ecuaciones base, qué se pretende y explicar que este tipo de
# experimentos son diseñados explícitamente, y que en el mundo de la inteligencia computacional este
# tipo de situaciones no son estándar...
#%%
# Importar los datos con pandas
data = pd.read_csv('datos-chidos.csv')
#%%
# Se pueden ver los datos
data
#%%
# Como faltan datos, se quitan las primeras y últimas dos filas
data = data.drop([0, 12])
#%%
# Revisar que todo esté bien
data
#%%
# Dado que sólo importan las columnas de t y v, se extraen de los datos
X = np.asarray(data['t'].values).reshape(-1, 1)
y = np.asarray(data['v_{y}'].values).astype(np.float)
print(X)
print(y)
#%%
# Graficar los datos para visualizar el dataset
plt.figure(figsize=(10, 6))
plt.scatter(X, y)
plt.xlabel(r'$t$')
plt.ylabel(r'$v_{y}$')
plt.show()
#%% [markdown]
# 2.1. _Primer_ intento, acercamiento _ingenuo_ a los datos.
#%%
# Visualizar el intervalo de confianza de una regresión lineal
plt.figure(figsize=(10,6))
sns.regplot(X, y)
#%%
from scipy.stats import linregress
slope, intercept, rvalue, __, __ = linregress(data['t'].values, y)
print(f'Slope: {slope}, Const: {intercept}, R^2: {rvalue}')
#%% [markdown]
# Estos datos pertenecen a la regresión ingenua...
#%% [markdown]
# Claramente es un modelo lineal, sin ruido, pero es un conjunto de datos muy pequeño
# y será difícil trabajar con este dataset de esta forma. ...
#%%
# Antes que nada, estandarizar los datos
scaler = MaxAbsScaler()
# Cambiar la forma del arreglo original para mantener consistencia
X = scaler.fit_transform(X)
print(X.shape)
#%% [markdown]
# Argumentar porqué se utiliza MaxAbsScaler, que es porque permite que lo valores constantes
# del modelo son más fáciles de regresar a su escala original...
#%%
print(y.shape)
#%%
def svr_fnc(x, x_tr=None, x_ts=None, y_tr=None, y_ts=None):
    # Crear una instancia del clasificador
    reg = SVR(kernel='linear', C=x[0], epsilon=x[1])
    # Ajustarlo con los datos de entrenamiento
    reg.fit(x_tr, y_tr)
    y_pred = reg.predict(x_ts)
    # Siempre se buscan valores positivos del accuracy
    score = np.abs(mean_squared_error(y_ts, y_pred))
    # score = reg.score(x_ts, y_ts)

    return -score
#%% [markdown]
# 2.2. _Bootstrap_ y el intervalo de confianza de múltiples muestreos
# Hablar sobre la técnica de bootstrap y si es posible, alguna imagen o referencia
#%%
# Opciones de bootstrap
n_iter = 100
n_size = int(len(X) * 0.50)
n_size
#%%
# Implementar bootstrap
stats = np.array([])
res_vals = np.zeros(2)
for i in range(n_iter):
    # Preparar los datos de prueba y entrenamiento
    x_train, y_train = resample(X, y, n_samples=n_size)
    x_test = np.array([x for x in X if x.tolist() not in x_train.tolist()])
    y_test = np.array([x for x in y if x.tolist() not in y_train.tolist()])
    # Ajustar el modelo
    # Estos son los parámetros de entrada del Firefly Algorithm
    kwargs = {'func': svr_fnc, 'dim': 2, 'tam_pob': 20, 'alpha': 0.9, 'beta': 0.2, 'gamma': 1.0, 
    'inf': 2**(-8), 'sup': 2**8}
    # Se crea una instancia del Firefly Algorithm
    fa_solve = fa.FAOpt(**kwargs, args=(x_train, x_test, y_train, y_test))
    # Se llama al método que resuelve la optimización
    res, fnc = fa_solve.optimizar(10, optim=True)

    # Se guardan los resultados de cada iteración
    res_vals += res
    # fnc pertenece a los errores de regresión encontrados
    stats = np.append(stats, fnc)
#%%
# Se estandarizan los datos según el número de muestras realizadas
res_vals /= n_iter
# Aquí se estandarizan los errores encontrados
stats /= n_iter
# Estos corresponden a los valores de C y epsilon óptimos
print(res_vals)
#%% [markdown]
# Mencionar distribución normal de errores (porqué debe de ser normal, qué tipo de hipótesis es,
# y la razón por la cual el número de muestras aumenta los valores de la distribución)...
#%%
# Usando una función especial de seaborn que muestra también la densidad de probabilidad
# de la distribución, mostrar la figura de los errores
plt.figure(figsize=(12,8))
sns.distplot(stats)
plt.savefig('errors.jpg')
plt.show()
#%%
# Usando percentiles y estadística descriptiva, se puede construir un intervalo de 
# confianza para la distribución de errores
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('{0:.1f} intervalo de confianza {1:.4f} y {2:.4f}'.format(alpha*100, lower, upper))
#%%
# Graficar los resultados obtenidos, se debería encontrar una gráfica ajustada
grv = SVR(kernel='linear', C=res_vals[0], epsilon=res_vals[1])
grv.fit(X, y)
print(grv.intercept_)
print(grv.coef_)
plt.figure(figsize=(10,6))
plt.scatter(X, y)
plt.plot(X, grv.fit(X, y).predict(X), c='r')
plt.xlabel(r'$t$')
plt.ylabel(r'$v_{y}$')
plt.show()
#%% [markdown]
#%% [markdown]
# No es un muy buen valor de epsilon, pues resulta en una función constante
# por lo que es necesario reducir el valor de epsilon
#%%
# Usando validación cruzada, se encuentra el nuevo valor de epsilon
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
grv = SVR(kernel='linear', C=res_vals[0])
params = {'epsilon': np.logspace(-8, 8, base=2.0)}
grv_cv = GridSearchCV(grv, param_grid=params, scoring='neg_mean_squared_error', cv=5, iid=False)
grv_cv.fit(x_train, y_train)
grv = grv_cv.best_estimator_
print(grv.intercept_)
print(grv.coef_)
print(grv.score(x_test, y_test))
#%%
# Y se grafican los resultados
plt.figure(figsize=(10,6))
plt.scatter(X, y)
plt.plot(X, grv.fit(X, y).predict(X), c='r')
plt.xlabel(r'$t$')
plt.ylabel(r'$v_{y}$')
plt.show()
#%% [markdown]
# Mencionar que el valor de epsilon es fundamental...
#%%
y_pred = grv.predict(X)
print(mean_squared_error(y, y_pred))
print(grv.fit(X, y).score(X, y))
#%%
print(grv.coef_/max(data['t'].values))
print(grv.intercept_/max(data['t'].values))

#%% [markdown]
# El modelo final es el siguiente:
# $$ y = -9.77245856 x + 0.63614353 $$
# y dado al intervalo de confianza realizado anteriormente, cada vez que se realice
# un ajuste lineal para este conjunto de datos, se podrá asegurar que el 95% de las veces
# se encontrará un para de coeficientes semejantes. Explicar más al respecto...
#%% [markdown]
# ## 3. Modelos _alternativos._
#%% [markdown]
# Hablar sobre Lasso (ecuaciones principales, explicación con alguna imagen si es posible)...
#%%
lasso_reg = LassoCV(eps=1e-4, n_alphas=300, cv=5, max_iter=3000)
lasso_reg.fit(x_train, y_train)
y_pred = lasso_reg.predict(x_test)
err = np.abs(mean_squared_error(y_test, y_pred))
print('MSE: {0}'.format(err))
print(lasso_reg.coef_/max(data['t'].values))
print(grv.intercept_/max(data['t'].values))
#%% [markdown]
# El modelo final es el siguiente:
# $$ y = -9.98784683 x + 0.63614338 $$
# y dado al intervalo de confianza realizado anteriormente, cada vez que se realice
# un ajuste lineal para este conjunto de datos, se podrá asegurar que el 95% de las veces
# se encontrará un para de coeficientes semejantes.
#%%
plt.figure(figsize=(10, 6))
plt.scatter(X, y)
plt.plot(X, lasso_reg.fit(X,y).predict(X))
plt.show()
#%% [markdown]
# ## Conclusiones
# 1. Hablar del número de datos y el problema que presenta.
# 2. Hablar de la técnica de muestreo de bootstrap y cómo se puede implementar en métodos de ensamble (muy rápido)
# 3. Hablar de las ventajas de utilizar Lasso para estimaciones rápidas y confiables
#%% [markdown]
# ## Referencias
# 
# 1. [Linear Regression in Machine Learning](https://www.ismll.uni-hildesheim.de/lehre/ml-07w/skript/ml-2up-01-linearregression.pdf)
# 2. [Regression and Stats Primer](http://polisci2.ucsd.edu/dhughes/teaching/OLS_Slides_Handout.pdf)
# 3. Citar el libro de Deng
# 4. pdf de Smolas
# 5. pdf de Burges
# 6. pdf de Basak