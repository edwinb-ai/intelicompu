# %% [markdown]
# # Funciones de Pérdida
# Comentarios sobre las funciones de pérdida...
# %%
import tensorflow as tf
import matplotlib.pyplot as plt

session = tf.Session()
# %% [markdown]
# ## 1. Regresión
# Mencionar sobre funciones de pérdida en regresión y porqué se necesitan...
# %%
# Crear algunos valores de prueba totalmente arbitrarios
x_pred = tf.linspace(-1.0, 1.0, 500)
x_real = tf.constant(0.0)
# %% [markdown]
# ### a. Norma en $L_2$ (Distancia Euclidiana)
# $$L_2 (y_r, y_p) = (y_r - y_p)^2$$
# Explicar cosas sobre la distancia Euclidiana...
# %%
# Implementar la función como tal está descrita
l2_y_vals = tf.square(x_real - x_pred)
# Mostrar solamente los primeros 20 valores
print(session.run(l2_y_vals[:20]))
# %% [markdown]
# La implementación de `TensorFlow` es muy específica, y se debe leer la
# [documentación](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/nn/l2_loss?hl=en)
# para comprender lo que hace. En particular esta función realiza la suma completa (o una suma reducida)
# y divide por 2 el resultado. A continuación se muestra la implementación y la función de `TensorFlow` como tal.
# %%
# Este es valor que viene implementado en TensorFlow
tf_l2 = tf.nn.l2_loss(x_real - x_pred)
print(session.run(tf_l2))
# %%
# Y esta es la implementación que equivale a la función de TensorFlow
l2_y_vals_tf = tf.reduce_sum(l2_y_vals) / 2.0
print(session.run(l2_y_vals_tf))
# %% [markdown]
# Y como se puede ver ambos resultados son iguales, pero esto es un buen ejemplo de que al escribir código siempre se
# debe tener a la mano la documentación oficial del paqueta o librería que se está empleando.
# %% [markdown]
# ### b. Norma en $L_1$ (Valor absoluto)
# $$L_1 (y_r, y_p) = \vert y_r - y_p \vert$$
# Explicar cosas sobre la distancia L1...
# %%
# Implementar la función como tal está descrita
l1_y_vals = tf.abs(x_real - x_pred)
# Mostrar solamente los primeros 20 valores
print(session.run(l2_y_vals[:20]))
# %% [markdown]
# ### c. Pseudo Huber
# $$L_{\delta} (y_r-y_p) = \delta^2 \cdot \sqrt{1 + \left( \frac{y_r-y_p}{\delta} \right) ^2} - 1$$
# Explicar cosas sobre la distancia Pseudo Huber, es una aproximación continue a Huber, etc...
# %%
# Se define un par de valorres arbitrario de delta, será diferente para cada conjunto de datos
delta_1 = tf.constant(0.25)
# Implementación de la función tal cual está definida
phuber1_y_vals = tf.multiply(tf.square(delta_1), tf.sqrt(
    1.0 + tf.square((x_real - x_pred)/delta_1)) - 1.0)
phuber1_y_out = session.run(phuber1_y_vals)
# Mostrar solo algunos valores del tensor
print(phuber1_y_out[:15])
print(session.run(tf.reduce_sum(phuber1_y_vals)))
# %%
# Esta es la implementación de TensorFlow que corresponde a la verdadera
# función de pérdida de Huber
tf_huber = tf.losses.huber_loss(tf.zeros_like(
    x_pred), x_pred, delta=delta_1, reduction=tf.losses.Reduction.NONE)
print(session.run(tf_huber[:15]))
#%% [markdown]
# Se puede ver que para valores de delta pequeños la función Pseudo Huber y la original
# tienen valores diferentes considerables, por lo que es importante siempre tratar de manejar
# en la medida de lo posible la función original.
#%% [markdown]
# Se vuelve a implementar este ejemplo, pero ahora con un valor diferente para delta.
# %%
delta_2 = tf.constant(5.0)
# Implementación de la función tal cual está definida
phuber2_y_vals = tf.multiply(tf.square(delta_2), tf.sqrt(
    1.0 + tf.square((x_real - x_pred)/delta_2)) - 1.0)
phuber2_y_out = session.run(phuber2_y_vals)
# Mostrar solo algunos valores del tensor
print(phuber2_y_out[:15])
print(session.run(tf.reduce_sum(phuber2_y_vals)))
# %%
tf_huber = tf.losses.huber_loss(tf.zeros_like(
    x_pred), x_pred, delta=delta_2, reduction=tf.losses.Reduction.NONE)
print(session.run(tf_huber[:15]))
#%% [markdown]
# Ahora se puede ver que los valores son casi los mismos, lo que implica que la función Pseudo Huber
# aproxima muy bien a la original para valores mayores a 1 y mantiene propiedades muy interesantes
# como _continuidad_ y _convexidad_.
#%% [markdown]
# ## 2. Clasificación
# Mencionar porqué se quiere una función de pérdidad en clasificación, algunas variedades, etc...
#%% [markdown]
# ### a. 