# %%
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# %% [markdown]
# ### Xavier and He Initialization
# %%
# Possible keras initializers:
[name for name in dir(keras.initializers) if not name.startswith("_")]

# %%
keras.initializers.he_normal()

# %%
# Keras dense layer with he initialization
layer = keras.layers.Dense(
    10,
    activation="relu",
    kernel_initializer="he_normal"
)

# %% [markdown]
# He initialization with a uniform distribution, but based on
# $fan_{avg}$ rather than $fan_{in}$, you can use the `VarianceScaling`
# initializer like this:
# %%
he_avg_init = keras.initializers.VarianceScaling(
    scale=2.,
    mode='fan_avg',
    distribution='uniform'
)
keras.layers.Dense(
    10,
    activation="sigmoid",
    kernel_initializer=he_avg_init
)

# %%
# Leaky ReLU
leaky_relu = keras.layers.LeakyReLU(alpha=0.2)
layer = keras.layers.Dense(
    10,
    activation=leaky_relu,
    kernel_initializer="he_normal"
)

# %%
# SELU activation
keras.layers.Dense(
    10,
    activation=keras.activations.selu,
    kernel_initializer="lecun_normal"
)

# %% [markdown]
# ### Batch Normalization with Keras
# %%
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])

# %%
model.summary()

# %%
[(var.name, var.trainable) for var in model.layers[1].variables]

# %%
# deprecated
model.layers[1].updates
# %%
# Batch normalization before the activation function
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(10, activation="softmax")
])

# %% [markdown]
# ### Pre-train a model and then reuse layers
# %%
# Create class names list and dictionary
class_names_list = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class_names_dict = dict(
    zip(class_names_list,
        range(len(class_names_list)))
)

class_names_list, class_names_dict

# %%
# Load the fashion mnist dataset
(X_train_full, y_train_full), (X_test,
                               y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize data
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Split X_train_full and y_train_full into training and validation sets
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# %% [markdown]
# Let's split the fashion MNIST training set in two
# * X_train_A: all images except for sandals and T-shirts (class 5 and 6)
# * X_train_B: 200 images of sandals and T-shirts
# %%


def split_dataset(X, y):
    y_5_or_6 = (y == 6) | (y == 5)
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2  # class indices 7, 8, 9 should be moved to 5, 6, 7
    y_B = (y[y_5_or_6] == 6).astype(np.float32)
    return ((X[~y_5_or_6], y_A), (X[y_5_or_6], y_B))


# %%
(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
(X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)

X_train_B = X_train_B[:200]
y_train_B = y_train_B[:200]

# %%
X_train_A.shape
# %%
X_train_B.shape
# %%
y_train_A[:30]
# %%
y_train_B[:30]
# %%
tf.random.set_seed(42)
np.random.seed(42)
# %%


def model_creator():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.BatchNormalization())
    for n_hidden in (300, 100, 50, 50, 50):
        model.add(keras.layers.Dense(n_hidden, activation="selu",
                  kernel_initializer="lecun_normal"))
        model.add(keras.layers.BatchNormalization())
    return model


# %%
model_A = model_creator()
model_A.add(keras.layers.Dense(8, activation="softmax"))
# %%
model_A.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                metrics=["accuracy"])
# %%
history = model_A.fit(X_train_A, y_train_A, epochs=20,
                      validation_data=(X_valid_A, y_valid_A))
# %%
model_A.save("my_model_A.h5")
# %%
model_B = model_creator()
model_B.add(keras.layers.Dense(1, activation="sigmoid"))
# %%
model_B.compile(loss="binary_crossentropy",
                optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                metrics=["accuracy"])
# %%
history = model_B.fit(X_train_B, y_train_B, epochs=20,
                      validation_data=(X_valid_B, y_valid_B))
# %%
model_B.summary()
# %%
# Load model_A and change the latest layer to sigmoid
# If we want to avoid to rewrite the model we can clone it
model_A = keras.models.load_model("my_model_A.h5")

model_A_cloned = keras.models.clone_model(model_A)
model_A_cloned.set_weights(model_A.get_weights())
model_B_on_A = keras.models.Sequential(model_A_cloned.layers[:-1])
model_B_on_A.add(
    keras.layers.Dense(
        1, activation="sigmoid", name="dense_13_sigmoid"
    )
)

# %%
# Make non-trainable all layers of model_B_on_A except the last one
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

model_B_on_A.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                     metrics=["accuracy"])
# %%
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=20,
                           validation_data=(X_valid_B, y_valid_B))
# %%
# Make trainable all layers of model_B_on_A
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

model_B_on_A.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                     metrics=["accuracy"])
# %%
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=20,
                           validation_data=(X_valid_B, y_valid_B))
# %%
# Check the final accuracy of the models
model_B.evaluate(X_test_B, y_test_B)
# %%
model_B_on_A.evaluate(X_test_B, y_test_B)
# %% [markdown]
# #### Fail, the model_B is better than the model_B_on_A
# %% [markdown]
# ## Faster Optimizers
# ### Momentum optimization
# %%
optimizer = keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
# %% [markdown]
# ### Nesterov momentum optimization
# %%
optimizer = keras.optimizers.SGD(
    learning_rate=1e-3,
    momentum=0.9,
    nesterov=True
)
# %%
# AdaGrad optimizer is not so good for deep neural networks
optimizer = keras.optimizers.Adagrad(learning_rate=0.001)
# %%
# RMSProp optimizer
optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
# %%
# Adam optimizer
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)
# %%
# Adamax optimizer
optimizer = keras.optimizers.Adamax(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)
# %%
# Nadam optimizer
optimizer = keras.optimizers.Nadam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)
# %% [markdown]
# ## Learning rate scheduling
# ### Power Scheduling
# %%
optimizer = keras.optimizers.SGD(
    learning_rate=0.01,
    decay=1e-4
)
# %%
# Create a model and check the PowerScheduler
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(
        300, activation="selu", kernel_initializer="lecun_normal"
    ),
    keras.layers.Dense(
        100, activation="selu", kernel_initializer="lecun_normal"
    ),
    keras.layers.Dense(
        50, activation="selu", kernel_initializer="lecun_normal"
    ),
    keras.layers.Dense(
        10, activation="softmax"
    )
])
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)
# %%
history = model.fit(
    X_train, y_train, epochs=25, validation_data=(X_valid, y_valid)
)
# %%
# Exponential Decay


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn


# %%
# Create a model and check the exponential_decay
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(
        300, activation="selu", kernel_initializer="lecun_normal"
    ),
    keras.layers.Dense(
        100, activation="selu", kernel_initializer="lecun_normal"
    ),
    keras.layers.Dense(
        10, activation="softmax"
    )
])
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="nadam",
    metrics=["accuracy"]
)
n_epoch = 25
# %%
lr_scheduler = keras.callbacks.LearningRateScheduler(
    exponential_decay(0.01, 20)
)
history = model.fit(
    X_train, y_train, epochs=n_epoch,
    validation_data=(X_valid, y_valid),
    callbacks=[lr_scheduler]
)
# %%
# Plot the learning rate vs epoch
plt.plot(history.epoch, history.history["lr"], 'o-')
plt.axis([0, n_epoch - 1, 0, 0.011])
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.title("Learning rate vs epoch")
plt.grid(True)
plt.show()
# %%
# Plot the loss vs epoch
plt.plot(history.epoch, history.history["loss"], 'o-')
plt.axis([0, n_epoch - 1, 0, 0.8])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs epoch")
plt.grid(True)
plt.show()
# %%
# Reduce the learning rate when the loss plateaus
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor="loss",
    factor=0.2,
    patience=2,
    verbose=1,
    mode="auto"
)
# %%
history = model.fit(
    X_train, y_train, epochs=n_epoch,
    validation_data=(X_valid, y_valid),
    callbacks=[lr_scheduler]
)
# %% [markdown]
# Keras scheduler
# %%
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(
        300, activation="selu", kernel_initializer="lecun_normal"
    ),
    keras.layers.Dense(
        100, activation="selu", kernel_initializer="lecun_normal"
    ),
    keras.layers.Dense(
        10, activation="softmax"
    )
])
# %%
n_epoch = 25
s = n_epoch * len(X_train) // 32

learning_rate = keras.optimizers.schedules.ExponentialDecay(
    0.01, s, 0.1
)
optimizers = keras.optimizers.SGD(learning_rate=learning_rate)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizers,
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train, epochs=n_epoch,
    validation_data=(X_valid, y_valid)
)
# %% [markdown]
# ## $\ell_1$ and $\ell_2$ regularization
# %% [markdown]
# How to apply $\ell_2$ regularization to the keras layer with factor 0.01
# %%
layer = keras.layers.Dense(
    100, activation="elu",
    kernel_initializer="he_normal",
    kernel_regularizer=keras.regularizers.l2(0.01))
# %%
# Create and train a model with $\ell_2$ regularization
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(
        300, activation="elu",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(0.01)
    ),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(
        100, activation="elu",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(0.01)
    ),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(
        10, activation="softmax",
        kernel_regularizer=keras.regularizers.l2(0.01)
    )
])
# %%
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizers,
    metrics=["accuracy"]
)
# %%
history = model.fit(
    X_train, y_train, epochs=n_epoch, validation_data=(X_valid, y_valid)
)
# %%
# Function params curry
RegularizedDense = partial(
    keras.layers.Dense,
    activation="elu",
    kernel_initializer="he_normal",
    kernel_regularizer=keras.regularizers.l2(0.01)
)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    RegularizedDense(300),
    RegularizedDense(100),
    RegularizedDense(10, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizers,
    metrics=["accuracy"]
)
# %%
n_epoch = 25
history = model.fit(
    X_train, y_train, epochs=n_epoch,
    validation_data=(X_valid, y_valid)
)
# %% [markdown]
# ## Dropout
# %%
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(
        300, activation="elu", kernel_initializer="he_normal"
    ),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(
        100, activation="elu", kernel_initializer="he_normal"
    ),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(
        10, activation="softmax"
    )
])
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="nadam",
    metrics=["accuracy"]
)
n_epoch = 2
history = model.fit(
    X_train, y_train, epochs=n_epoch, validation_data=(X_valid, y_valid)
)
# %%
# Alpha dropout if you want to use selu activation
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(
        300, activation="selu", kernel_initializer="lecun_normal"
    ),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(
        100, activation="selu", kernel_initializer="lecun_normal"
    ),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(
        10, activation="softmax"
    )
])

optimizer = keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.9, nesterov=True
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)

n_epoch = 10

history = model.fit(
    X_train, y_train, epochs=n_epoch, validation_data=(X_valid, y_valid)
)
# %%
model.evaluate(X_test, y_test)
# %% [markdown]
# ## Monte Carlo Dropout
# %%
y_probas = np.stack([model(X_test, training=True) for _ in range(100)])

y_proba = np.mean(y_probas, axis=0)
y_std = np.std(y_probas, axis=0)
# %%
np.round(model.predict(X_test[:1]), 2)
# %%
np.round(y_probas[:, :1], 2)
# %%
np.round(y_proba[:1], 2)
# %%
np.round(y_std[:1], 2)
# %%
# Boost of accuracy!
y_pred = np.argmax(y_proba, axis=1)
accuracy = np.mean(y_pred == y_test)
accuracy
# %%
# Let's create a class MCDropout that implements the Monte Carlo dropout


class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


# %%
mc_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    MCDropout(rate=0.2),
    keras.layers.Dense(
        300, activation="elu", kernel_initializer="he_normal"
    ),
    MCDropout(rate=0.2),
    keras.layers.Dense(
        100, activation="elu", kernel_initializer="he_normal"
    ),
    MCDropout(rate=0.2),
    keras.layers.Dense(
        10, activation="softmax"
    )
])

mc_model.summary()
# %%
optimizer = keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.9, nesterov=True
)
mc_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)
# %%
mc_model.set_weights(model.get_weights())
# %%
np.round(
    np.mean(
        [mc_model.predict(X_test[:1]) for _ in range(100)], axis=0
    ), 2
)
# %% [markdown]
# ## Max Norm Regularization
# %%
layer = keras.layers.Dense(
    100, activation="elu", kernel_initializer="he_normal",
    kernel_constraint=keras.constraints.MaxNorm(max_value=1.0)
)
# %%
