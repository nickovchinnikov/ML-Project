# %%
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler

from scipy.stats import reciprocal

import tensorflow as tf
from tensorflow import keras

# %%

iris = load_iris()

X = iris.data[:, (2, 3)]  # petal length, petal width
y = (iris.target == 0).astype(np.int32) # Iris-setosa?

per_clf = Perceptron()
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])
y_pred

# %%
tf.__version__

# %%
keras.__version__

# %% [markdown]
# ### Building an Image Classifier Using the Sequential API
# %%
fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# %%
X_train_full.shape

# %%
y_train_full.dtype

# %%
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_test = X_test / 255.

# %%
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class_names[y_train[0]]
# %%
plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()

# %%
X_valid.shape, X_test.shape

# %%
# %%
# Let's check the images in the dataset
n_rows = 4
n_cols = 10

plt.figure(figsize=(n_cols * 2, n_rows * 2))

for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]], fontsize=12)

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# %%
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()

# %%
# Insted of add it one by one, we can use the Sequential API
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()

# %%
# list of layers
model.layers, model.layers[0].name

# %%
model.get_layer('dense_3').name

# %%
# Plot the model need to install pydot and graphviz
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

# %%
hidden1 = model.layers[1]
hidden1.name, hidden1.output_shape, hidden1.input_shape

# %%
weights, biases = hidden1.get_weights()

# %%
# Weights initalized with random values, to break symmetry
weights

# %%
biases

# %%
# Compile the model
model.compile(loss="sparse_categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"])

# %%
# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_data=(X_valid, y_valid)
)

# %%
history.params

# %%
history.epoch

# %%
history.history.keys()

# %%
# Plot the history of the model
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# %%
# Evaluate the model
model.evaluate(X_test, y_test)

# %%
# Make predictions
X_new = X_test[:3]

y_proba = model.predict(X_new)
y_proba.round(2)

# %%
# Predict the class
# Warning model.predict_classes is deprecated!
#y_pred = model.predict_classes(X_new)
y_pred = np.argmax(model.predict(X_new), axis=-1)
y_pred

# %%
np.array(class_names)[y_pred]

# %% [markdown]
# Building a regression MLP
# %%
housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# %%
X_train.shape, X_test.shape, X_valid.shape

# %%
model = keras.models.Sequential([
    keras.layers.Dense(
        30,
        activation="relu",
        input_shape=(X_train.shape[1:])
    ),
    keras.layers.Dense(1)
])

model.compile(loss=keras.losses.mse, optimizer="sgd")

# %%
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_valid, y_valid)
)

# %%
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)

# %%
plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# %%
y_pred

# %% [markdown]
# ### Complex model with Functional API
# %%
input = keras.layers.Input(shape=(X_train.shape[1:]))
hidden1 = keras.layers.Dense(30, activation="relu")(input)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([hidden1, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input], outputs=[output])

# %%
# Complete the model
model.compile(loss=keras.losses.mse, optimizer="sgd")

# %%
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_valid, y_valid)
)
# %%
y_pred = model.predict(X_new)
y_pred
# %% [markdown]
# For example, suppose we want to send 5 features through
# the deep path (features 0 to 4), and 6 features through
# the wide path (features 2 to 7):

# %%
input_A = keras.layers.Input(shape=(5))
input_B = keras.layers.Input(shape=(6))

hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)

concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.Model(inputs=[input_A, input_B], outputs=[output])

# %%
# Complete the model
model.compile(loss=keras.losses.mse, optimizer="sgd")

# %%
X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

# %%
# Fit the model
history = model.fit(
    (X_train_A, X_train_B), y_train, epochs=20,
    validation_data=((X_valid_A, X_valid_B), y_valid)
)

# %%
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))

mse_test, y_pred
# %%
# Multiple outputs model
input_A = keras.layers.Input(shape=(5))
input_B = keras.layers.Input(shape=(6))

hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)

concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1)(concat)

aux_output = keras.layers.Dense(1)(hidden2)

model = keras.Model(
    inputs=[input_A, input_B],
    outputs=[output, aux_output]
)

model.compile(
    loss=[keras.losses.mse, keras.losses.mse],
    loss_weights=[0.9, 0.1],
    optimizer="sgd"
)

# %%
# Pass the data to the model
history = model.fit(
    (X_train_A, X_train_B), [y_train, y_train], epochs=20,
    validation_data=((X_valid_A, X_valid_B), [y_valid, y_valid])
)

# %%
# Let's check the loss and the accuracy on the test set
total_loss, mse_loss, aux_loss = model.evaluate(
    (X_test_A, X_test_B), [y_test, y_test]
)

# %%
# Predict the values
y_pred_main, y_pred_aux = model.predict((X_new_A, X_new_B))
# %%
y_pred_main, y_pred_aux
# %% [markdown]
# ### Subclassing API
# %%
class WideAndDeepModel(keras.models.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel()

# %%
model.compile(
    loss=[keras.losses.mse, keras.losses.mse],
    loss_weights=[0.9, 0.1],
    optimizer="sgd"
) 

# %%
# Pass the data to the model
history = model.fit(
    (X_train_A, X_train_B), [y_train, y_train], epochs=20,
    validation_data=((X_valid_A, X_valid_B), [y_valid, y_valid])
)

# %%
# Let's check the loss and the accuracy on the test set
total_loss, mse_loss, aux_loss = model.evaluate(
    (X_test_A, X_test_B), [y_test, y_test]
)

total_loss, mse_loss, aux_loss

# %%
# Predict the values
y_pred_main, y_pred_aux = model.predict((X_new_A, X_new_B))
# %%
y_pred_main, y_pred_aux

# %% [markdown]
# ### Saving and restoring a Model
# %%
model.save("my_model.tf")

# %%
loaded_model = keras.models.load_model("my_model.tf")

# %% [markdown]
# ### Using callbacks
# %%
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "my_model.tf", save_best_only=True
)

history = model.fit(
    (X_train_A, X_train_B), [y_train, y_train], epochs=10,
    validation_data=((X_valid_A, X_valid_B), [y_valid, y_valid]),
    callbacks=[checkpoint_cb]
)

loaded_model = keras.models.load_model("my_model.tf")

# %%
# Early stop callback
early_stop_cb = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    (X_train_A, X_train_B), [y_train, y_train], epochs=100,
    validation_data=((X_valid_A, X_valid_B), [y_valid, y_valid]),
    callbacks=[checkpoint_cb, early_stop_cb]
)

# %%
# Custom callback that calculate the ratio between
# the validation loss and the training loss
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"\nValidation/Train loss ratio: {logs['val_loss'] / logs['loss']}"
        )

# %%
# Pass the callback to the model
history = model.fit(
    (X_train_A, X_train_B), [y_train, y_train], epochs=100,
    validation_data=((X_valid_A, X_valid_B), [y_valid, y_valid]),
    callbacks=[checkpoint_cb, early_stop_cb, PrintValTrainRatioCallback()]
)

# %% [markdown]
# ### Visualization with TensorBoard
# %%
root_logdir = os.path.join(os.curdir, "logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

# %%
tensorboard_cb = keras.callbacks.TensorBoard(
    run_logdir,
)
# %%
history = model.fit(
    (X_train_A, X_train_B), [y_train, y_train], epochs=20,
    validation_data=((X_valid_A, X_valid_B), [y_valid, y_valid]),
    callbacks=[tensorboard_cb]
)

# %% [markdown]
# ### Fine-Tuning Neural Networks Hyperparameters
# %%
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))

    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(loss="mse", optimizer=optimizer)
    return model

# %%
# Create a KerosRegressor based on the model
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

# %%
keras_reg.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_valid, y_valid),
    callbacks=[
        tensorboard_cb,
        keras.callbacks.EarlyStopping(patience=10)
    ]
)

# %%
mse_test = keras_reg.score(X_test, y_test)
mse_test

# %%
X_new = X_test[:3]
y_pred = keras_reg.predict(X_new)
y_pred
# %%
# Use GridSearchCV to find the best parameters
param_distribs = {
    "n_hidden": [1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(
    keras_reg,
    param_distribs,
    n_iter=10,
    cv=3

)
rnd_search_cv.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_valid, y_valid),
    callbacks=[keras.callbacks.EarlyStopping(patience=10)]
)

# %%
rnd_search_cv.best_params_

# %%
rnd_search_cv.best_score_

# %%
rnd_search_cv.best_model_.save("best_model.tf")

# %%
# Save the best model
import joblib

joblib.dump(rnd_search_cv, "best_model.pkl")
# %%
