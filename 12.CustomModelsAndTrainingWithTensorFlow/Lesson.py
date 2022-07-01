# %%
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import tensorflow.keras as keras

# %%
# Matrix with 2 rows and 3 columns
t = tf.constant([
    [1., 2., 3.],
    [4., 5., 6.]
])
t
# %%
# Scalar
tf.constant(42)
# %%
t.shape, t.dtype
# %%
# Indexing work much like in numpy
t[:, 1:]
# %% [markdown]
# #### All sorts of tensor operations are possible
# %%
# Elementwise sum of all elements
t + t
# %%
# Add 100 to each element
t + 100
# %%
# The same as above
tf.add(t, 100)
# %%
# Square of each element
tf.square(t)
# %%
# Matrix multiplication
t @ tf.transpose(t)
# %% [markdown]
# ### Tensors and NumPy arrays
# %%
a = np.array([2, 4, 5])

t = tf.constant(a)
t
# %%
# Convert a tensor to a numpy array
t.numpy()
# %%
tf.square(a)
# %%
np.square(t)
# %% [markdown]
# ### Types conversion
# %%
# Error, tf can't convert a data type automatically
try:
    tf.constant(2.) + tf.constant(40)
except tf.errors.InvalidArgumentError as err:
    print(err)
# %%
# Can't even add tf.float64 instead of tf.float32
try:
    tf.constant(2.) + tf.constant(40, dtype=tf.float64)
except tf.errors.InvalidArgumentError as err:
    print(err)
# %%
# You can use tf.cast to convert data types
t2 = tf.constant(a, dtype=tf.float64)

# No errors!
tf.constant(2.) + tf.cast(t2, tf.float32)
# %% [markdown]
# ### Variables
# %%
v = tf.Variable([
    [1., 2., 3.],
    [4., 5., 6.]
])
v
# %%
# You can change a variable with assign operation
v.assign(2 * v)
# %%
v[0, 1].assign(42)
# %%
v[:, 2].assign([0., 1.])
# %%
v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.])
# %% [markdown]
# ### Customizing Models and Training Algorithms
# #### Custom loss function
# %%
# Fetch the California housing data
housing = fetch_california_housing()
(data, target) = housing.data, housing.target.reshape(-1, 1)

(data[:1], data.shape), (target[:2], target.shape)
# %%
X_train_full, X_test, y_train_full, y_test = train_test_split(
    data,
    target,
    random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    random_state=42
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# %%
# Huber loss


def huber_loss(y_true, y_pred, delta=1.):
    # Cast to float32
    error = tf.cast(y_true, dtype=tf.float32) - \
        tf.cast(y_pred, dtype=tf.float32)
    abs_error = tf.abs(error)
    is_small_error = abs_error < delta
    squared_loss = tf.square(error) / 2.
    linear_loss = delta * abs_error - tf.square(delta) / 2.
    return tf.where(
        is_small_error,
        squared_loss,
        linear_loss
    )


# %%
# Let's plot the huber loss
plt.figure(figsize=(10, 4))
z = np.linspace(-4, 4, 200)

# plot the huber loss
plt.plot(z, huber_loss(0, z), 'b-', linewidth=2, label=r"huber($z$)")
plt.plot(z, z ** 2 / 2, 'b:', linewidth=1, label=r"$\frac{1}{2}z^2$")

plt.plot([-1, -1], [0, huber_loss(0., -1.)], "r--")
plt.plot([1, 1], [0, huber_loss(0., 1.)], "r--")

plt.gca().axhline(y=0, color='k')
plt.gca().axvline(x=0, color='k')

plt.axis([-4, 4, 0, 4])
plt.grid(True)

plt.legend(fontsize=14)
plt.title("Huber loss", fontsize=14)
plt.show()
# %%
# Create a model with huber loss
input_shape = X_train.shape[1:]

model = keras.models.Sequential([
    keras.layers.Dense(
        30,
        activation='selu',
        kernel_initializer='lecun_normal',
        input_shape=input_shape
    ),
    keras.layers.Dense(1),
])

model.compile(
    optimizer='nadam',
    loss=huber_loss,
    metrics=['mse']
)
# %%
# Train the model
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=2,
    validation_data=(X_valid_scaled, y_valid),
)
# %% [markdown]
# ### Saving and Loading Models That Contain Custom Components
# %%
# Save the model
model.save("my_model_with_a_custom_loss.h5")
# %%
# Load the model
model = keras.models.load_model(
    "my_model_with_a_custom_loss.h5",
    custom_objects={'huber_loss': huber_loss}
)
# %%
huber_loss2 = partial(huber_loss, delta=2.)

model.compile(
    optimizer='nadam',
    loss=huber_loss2,
    metrics=['mse']
)
# %%
model.fit(
    X_train_scaled,
    y_train,
    epochs=2,
    validation_data=(X_valid_scaled, y_valid),
)
# %%
# Save the model
model.save("my_model_with_a_custom_loss_threshold_2.h5")
# %%
# Load the model
model = keras.models.load_model(
    "my_model_with_a_custom_loss_threshold_2.h5",
    custom_objects={'huber_loss': huber_loss2}
)
# %% [markdown]
# #### Other Custom Functions
# %%
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
# %%


def my_softplus(z):  # return value is just tf.nn.softplus(z)
    return tf.math.log(tf.exp(z) + 1.0)


def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)


def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))


def my_positive_weights(weights):  # return value is just tf.nn.relu(weights)
    return tf.where(weights < 0., tf.zeros_like(weights), weights)


# %%
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1, activation=my_softplus,
                       kernel_regularizer=my_l1_regularizer,
                       kernel_constraint=my_positive_weights,
                       kernel_initializer=my_glorot_initializer),
])
model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
# %%
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
# %%
# Save the model
model.save("my_model_with_custom_components.h5")
# %%
# Load the model
model = keras.models.load_model(
    "my_model_with_custom_components.h5",
    custom_objects={
        "my_softplus": my_softplus,
        "my_glorot_initializer": my_glorot_initializer,
        "my_l1_regularizer": my_l1_regularizer,
        "my_positive_weights": my_positive_weights
    }
)
# %% [markdown]
# ### Custom Metrix
# %%
# Keras precision metric
precision = keras.metrics.Precision()
precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
# %%
precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])
# %%
precision.result()
# %%
precision.variables
# %%
# Reset state of the metric
precision.reset_states()
# %% [markdown]
# ### Custom Layer
# %%
# Exponential Layer
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
# %%
