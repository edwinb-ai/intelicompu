#%% [markdown]
# # _Regresión simple_ (con Máquinas de Soporte Vectorial).

#%% [markdown]
# ## 1. Introducción
# La _regresión lineal_ es un concepto familiar de la estadística paramétrica donde se aplican muchos de los principios y teoremas
# fundamentales de esta área. Existen diversos propósitos para realizar una regresión, pero en particular se pueden reducir a dos
# principales:
#
# 1. _Ajustar_ la mejor linea recta a un conjunto de datos, y con esto se cumplen dos subobjetivos:
# 
# a. Encontrar los _parámetros_ de la mejor linea recta. Estos pueden ser lo que se busca originalmente en algún tipo de experimento diseñado.
# 
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
# $$ RSS = \sum_{i=1}^{N} \hat{\varepsilon_i}^2 =  \sum_{i=1}^{N} \left(y_i - \left(\alpha + \beta x_i \right)\right)^2 ,$$
# donde $N$ es el número total de valores tanto en $\mathbf{X}$ como en $\mathbf{Y}$ y $\hat{\varepsilon_i}$ son los errores de _predicción_ respecto a los
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
# 1. El **valor esperado** de los errores $\varepsilon$ debe cumplir que $E\left(\varepsilon \vert x\right) = 0.$
# 2. La **varianza** de los errores debe de ser _finita_, _constante_ y _conocida_ tal que cumpla $V\left(\varepsilon \vert x\right) = \sigma^2 < \infty.$
# 3. **No** existe una _correlación_ entre los errores tal que $Cov\left(\varepsilon_i \vert \varepsilon_j\right) = 0.$
#
#
# Lo más _común_ es asumir que los _errores_ siguen una distribución _normal_ tal que $\varepsilon \vert x \sim \mathcal{N}\left(0, \sigma^2\right)$
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
# La formulación estándar de las SVM para regresión se conoce como $\varepsilon-SVR$ (epsilon support vector regression, en inglés). Para realizar la formulación
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
# y_i - \langle \omega, x_i \rangle - b \leq \varepsilon, \\[2ex]
# \langle \omega, x_i \rangle + b - y_i \leq \varepsilon .
# \end{cases}
# \end{aligned}
# $$
# 
# Este problema es en realidad un problema de _optimización convexa_ que es _viable_ cuando $f$ existe y aproxima todo los pares $(x_n,y_n)$ con precisión
# $\varepsilon.$ Sin embargo, como en el caso de SVM de margen suave, existe una versión análoga para el caso de SVR donde se introducen dos variables
# que permitan un cierto margen de error, por lo que se puede reformular todo el problema de optimización anterior por el siguiente:
# $$
# \begin{aligned}
# & \text{minimizar} \quad
# \frac{1}{2} \vert \vert \omega \vert \vert^2 + C \sum_{i=1}^{N} (\xi_i - \xi_i^{\star}) \\[2ex]
# & \text{sujeto a }
# \begin{cases}
# y_i - \langle \omega, x_i \rangle - b &\leq \varepsilon + \xi_i, \\[2ex]
# \langle \omega, x_i \rangle + b - y_i &\leq \varepsilon + \xi_i^{\star}, \\[2ex]
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
# y $k(x_i, x) = \varphi(x_i) \varphi(x)$ es un _kernel de Mercer_ (revisar la referencia 4 para más información).
# ¿Cómo se determinan? Claro está que sin estos valores **no** se puede
# encontrar el modelo lineal buscado.
#
# ### Función de riesgo
# NOTA: **No** se llevarán a cabo las demostraciones en esta libreta, y la presentación de las ecuaciones será meramente ilustrativa.
# Para aquella persona que desee conocer más al respecto, leer con detalle la referencia 3 y 6.
#
# Continuando con el argumento anterior, dado que $\omega$ está escrito en función de $\alpha_i - \alpha_i^{\star}$ se puede entonces determinar estos
# parámetros en términos de $\omega$ usando la siguiente [_funcional de riesgo_](https://en.wikipedia.org/wiki/Statistical_risk):
# $$ R_{reg}[f] = \frac{1}{2} \vert \vert \omega \vert \vert^2 + C \sum_{i=1}^{N} L_{\varepsilon}(y) ,$$
# donde $L_{\varepsilon}(y)$ es la [_función de pérdida_](https://en.wikipedia.org/wiki/Loss_function) llamada _$\varepsilon$-insensitive loss function_ definida
# originalmente por Vladimir Vapnik en la formulación original de este algoritmo como sigue:
# $$ 
# L_{\varepsilon}(y) = 
# \begin{cases}
# \qquad 0 , & \quad \text{para}\ \vert f(x) - y \vert < \varepsilon, \\[2ex]
# \vert f(x) - y \vert - \varepsilon , & \quad \text{cualquier otro caso.}
# \end{cases}
# $$
# 
# Como se puede observar, este problema de optimización es complejo y difícil de resolver con métodos tradicionales. En esta libreta se empleará la librería estándar
# [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) que viene en la librería de Python _scikit-learn_ para realizar los ejemplos a continuación.
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
import os
# Cargar el Firefly Algorithm, la metaheurística para encontrar los hiperparámetros adecuados
# NOTA: Este comando carga todos los contenidos de firefly, usarlo bajo discreción
firefly_arch = os.path.abspath('metaheuristicas/firefly/firefly.py')
%run $firefly_arch

#%% [markdown]
# ## 1. Complejidad del modelo (sobreajuste)
# En esta primera sección se implementa una regresión utilizando SVM aplicadas para el problema de
# regresión lineal. Sin embargo, como se ha visto en la teoría, las SVM pueden tomar una _función kernel_ para realizar un _mapeo_ de los datos
# originales a otro espacio para realizar el ajuste lineal. Sin embargo, cuando los datos son _lineales por naturaleza_ emplear un _kernel_
# puede provocar un **sobreajuste** del modelo.
# 
# ### 1.1. Sobreajuste
# El _sobreajuste_ sucede cuando un modelo aprende muchas de las características base del conjunto de datos de entranamiento. Esto implica que
# cuando se aplica el método a datos diferentes el modelo _no_ puede diferenciar correctamente entre datos que **son** una características y 
# datos que **no lo son.** Esto es fundamental, sobre todo en las SVM. ¿Porqué?
#
# Cuando se busca emplear las SVM se pretende que el número de vectores soporte (revisar referencia 3) sea muy pequeño tal que con un número limitado
# de datos el modelo tenga un buen ajuste; esta es una característica y ventaja de la naturaleza de las SVM. Sin embargo, cuando el modelo empieza a tener
# una **alta complejidad**, i.e. que tenga muchos hiperparámetros por determinar, el modelo describe a los datos de una forma errónea a su verdadera
# correlación. Una forma fácil de identificar el sobreajuste en un modelo de SVM es que el número de vectores soporte es muy grande, llegando al límite
# de que cada dato dentro del conjunto de datos es un vector soporte. Esto se estudiará con más detalle en el ejemplo a continuación.
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
# Como se esperaba el modelo es lineal pero algo importante de notar es la implementación de _scikit-learn_ donde se tuvo que especificar
# un parámetro de _ruido_ y un parámetro de _bias_. En particular esto se tiene que hacer dado que la implementación crearía un modelo lineal
# perfecto, y se busca que el modelo tenga al menos una cierta _dispersión_ en los datos.
#%%
# Separar el conjunto de datos en prueba y entrenamiento
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
#%% [markdown]
# ## 1.2. Kernel _lineal_
# Esta implementación de la SVM corresponde al modelo lineal que se presentó en la sección de teoría. Para este caso, se espera que este kernel
# sea el mejor y que tenga una muy buena representación respecto a la correlación de los datos. Si llegara a tener _sobreajuste_ se puede deber
# a la mala optimización o ajuste de los hiperparámetros.
#%%
# Crear un regresor con kernel lineal
lineal = SVR(kernel='linear')
# Definir parámetros para la validación cruzada
params = {'C': sp_int(2**(-15), 2**15), 'epsilon': sp_int(2**(-8), 2**8)}
# Realizar la validación cruzada aleatoria
reg_lin = RandomizedSearchCV(lineal, param_distributions=params, scoring='neg_mean_squared_error',
n_iter=25, cv=10, iid=False, n_jobs=-1)
# Ajustar el modelo
reg_lin.fit(x_train, y_train)
#%% [markdown]
# ### Una nota sobre RandomizedSearchCV
# En esta parte del código se empleó la función [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
# para realizar validación cruzada del kernel lineal. Esto se realizó dado que el rango de búsqueda para los hiperparámetros es muy grande, y el método
# usual de Grid Search tardaría demasiado tiempo.
# En particular, _Randomized Search Cross Validation_ emplea el uso de distribuciones de probabilidad para realizar un muestreo del espacio de
# búsqueda y crear un conjunto óptimo de hiperparámetros. Es importante notar que esto es _aleatorio_ y que **no** se busca el espacio completo
# por lo que es posible que no siempre se obtengan los mismos resultados. Después en esta libreta se empleará otro método mucho más eficiente
# pero se deja esta por referencia de que también es una opción viable y rápida para obtener casi los mismos resultados que Grid Search.
#%%
# Mostrar los méjores parámetros
lineal = reg_lin.best_estimator_
print(lineal)

# Mostrar el R^2
print(lineal.score(x_train, y_train))
#%% [markdown]
# Aunque aquí estamos mostrando el valor de correlación $R^2$ este valor **no** es una buena métrica para este tipo de modelos. Este
# valor numérico sólo corresponde a la _correlación_ entre los datos, pero en particular nos importa _minimizar_ el error entre
# la predicción y los valores reales del conjunto de datos. Por lo tanto, es muy común (y se emplea en esta libreta) el uso de otro
# tipo de métricas, en especial el [error cuadrático medio](https://en.wikipedia.org/wiki/Mean_squared_error), definido matemáticamente
# como:
# $$ MSE = \frac{1}{N} \sum_{i=1}^{N} (Y_i - \hat{Y}_i)^2 ,$$
# donde $Y$ son los datos reales y $\hat{Y}$ son los datos predichos por el modelo.
#%%
# Con este regresor, realizar predicciones
y_pred = lineal.predict(x_test)
# Mostrar solo algunos valores para visualización
y_pred[:5]
#%%
# Calcular el error entre entrenamiento y prueba, se emplea la raíz cuadrada dado que MSE puede
# ser un número muy grande
print('RMSE para entrenamiento: {0}'.format(np.sqrt(np.abs(mean_squared_error(y_train, lineal.predict(x_train))))))
print('RMSE para prueba: {0}'.format(np.sqrt(np.abs(mean_squared_error(y_test, y_pred)))))
#%% [markdown]
# Dado que los errores son muy semejantes, **NO** existe sobreajuste del modelo. ¿A qué se debe esto? Es relativamente simple de ver.
# El error cuadrático medio es una medida de el desempeño del modelo para predecir los valores y la comparación con los datos
# originales, entonces, cuando el error es muy grande significa que el modelo **no predice** correctamente, no tiene la información
# suficiente de la _correlación_ entre los datos.
# 
# En particular, cuando el modelo está _sobreajustado_ significa que ha tomado en cuenta muchas de las características de un subconjunto
# de los datos que **no** están presentes en todo el conjunto de datos por completo. De esta forma, el error será muy grande en un parte
# del proceso (e.g. la predicción del modelo) y será muy pequeño en la otra parte del proceso (e.g. el entrenamiento del modelo) debido
# a que no existe una **generalización** del modelo para otro subconjunto de datos dentro del mismo conjunto de datos total.
#
# Sin embargo, en este caso los errores medidos son relativamente _cercanos_ entre sí, lo que significa que el modelo puede **generalizar**
# efectivamente la _correlación_ de los datos, prediciendo correctamente los datos dentro de un margen de error.
#%% [markdown]
# ## 1.2. Kernel _base radial._
# El kernel de _base radial_ se define como
# $$ K(\mathbf{x},\mathbf{x'}) = \exp{ \left( - \gamma \vert \vert \mathbf{x} - \mathbf{x'} \vert \vert^2 \right)}$$
# y es importante notar que cuando se tiene el límite $\gamma \to 0$ entonces se recupera el _modelo lineal_ dado que
# $$ \lim_{\gamma \to 0} \exp{ \left( - \gamma \vert \vert \mathbf{x} - \mathbf{x'} \vert \vert^2 \right)} = 1$$
# y usando la descripción presentada al principio de esta libreta, entonces el modelo se convierte a una función linea, recuperando entonces
# la regresión lineal con SVM.
# 
# De esta forma se espera que el valor de $\gamma$ que se encuentre sea pequeño, cercano a 0. Sin embargo, aunque esto se logre, este _kernel_
# se espera que tenga un bajo desempeño dado que tiene que mapear los datos a un espacio diferente y ahí realizar la separación lineal. Esto
# _no_ está garantizado y por tanto se _espera_ que exista un alto número de vectores soporte que sean empleados para crear el modelo.
#%% [markdown]
# ## Sección _bonus_. _Metaheurísticas._
# En esta sección se menciona rápidamente un método para realizar el ajuste de los hiperparámetros. En particular se desea minimizar el error
# de predicción y esto se puede traducir a la elección correcta de hiperparámetros que emplea el modelo tal que sean los mejores que efectivamente
# _minimizan_ algún tipo de métrica como el MSE.
#
# Las **metaheurísticas** son métodos de _optimización_ muy robustos que permiten la resolver problemas de optimización en muchas dimensiones, para
# funciones multimodales entre otras cosas.
#
# En particular, en esta libreta se emplean para el ajuste de hiperparámetros de las SVM, pero dado que no es el propósito de esta libreta realizar
# un estudio detallado y sistemático de estos algoritmos, sólo se menciona su mecánica general y en otra libreta se hará este estudio.
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
# Este arreglo guarda los parámetros óptimos; gamma, C y epsilon
res_vals = np.zeros(3)
# Este arreglo guarda el valor de accuracy total
fnc_total = np.array([])

# Se comienza a iterar sobre los 10 pliegues de la validación cruzada
for tr, ts in skf.split(x_train, y_train):
    # Estos son los parámetros de entrada del Firefly Algorithm
    kwargs = {'func': svr_fnc, 'dim': 3, 'tam_pob': 20, 'alpha': 0.9, 'beta': 0.2, 'gamma': 1.0, 
    'inf': 2**(-4), 'sup': 2**4}
    # Se crea una instancia del Firefly Algorithm
    fa_solve = FAOpt(**kwargs, args=(X[tr], X[ts], y[tr], y[ts]))
    # Se llama al método que resuelve la optimización
    res, fnc = fa_solve.optimizar(15, optim=True)
    
    # Se guardan los resultados de cada iteración
    res_vals += res
    fnc_total = np.append(fnc_total, fnc)

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
# Ahora bien, la variación de los errores encontrados _no_ es considerablemente grande, pero lo es. Esto puede significar que existe sobreajuste
# como se había conjeturado, pero no es una buena forma de comparación. Como se había comentado, otra forma útil de corroborar este hecho
# es encontrando el número de vectores soporte de cada modelo; entre mayor sea el modelo, es más complejo y por tanto puede existir un sobreajuste.
#%%
# Código tomado de
# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py
# y adaptado para esta libreta.
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
#
# 1. Como se puede ver, existe un número mayor de vectores soporte para el kernel _rbf_, que era de esperarse. La figura que crea la gráfica puede
# mejorarse si el valor de $\gamma$ es más pequeño, pero aquí lo importante es el número de vectores soporte. Claramente este caso es lineal
# y debe utilizarse un kernel _lineal_ para este problema.
#
# 2. La parte importante de crear estos modelos de inteligencia computacional para _regresión_ es la **predicción**. ¿Qué tan bien predicen
# estos modelos? Por el _análisis_ de los errores encontrados y los coeficientes de correlación, claramente el kernel _lineal_ es el mejor
# pero se puede argumentar que dentro de un margen de error el kernel _rbf_ es aceptable. 
#
# 3. Por último, siempre es preferible un modelo que tenga una **menor complejidad**, i.e. el número de hiperparámetros es pequeño. Esto porque
# facilita la búsqueda de estos valores, es más eficiente y se puede evitar en gran medida el problema de _sobreajuste._ En dado caso que
# el problema lo requiera y se necesite un kernel de tipo _rbf_ se puede emplear una _metaheurística_ para acelerar considerablemente la búsqueda
# de los hiperparámetros.
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
    score = np.sqrt(np.abs(mean_squared_error(y_ts, y_pred)))
    # score = reg.score(x_ts, y_ts)

    return score
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
    fa_solve = FAOpt(**kwargs, args=(x_train, x_test, y_train, y_test))
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
# 7. [Asymptotic Behaviors of Support Vector Machines with Gaussian Kernel](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.141.880&rep=rep1&type=pdf)