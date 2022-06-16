# %% [markdown]
# ## Exercise 10: Deep Learning Intro
# Train a Deep Neural Network on MNIST dataset \
# Can I make more than 98% accuracy?
# %%
# Split the data into training and test sets
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

# %%
# Check the shape of the data and data type
X_train_full.shape, X_train_full.dtype

# %%
# Split the training data into training and validation sets
# And scale the pixel values to the range [0, 1]
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
X_test = X_test / 255.0

y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# %%
plt.imshow(X_train[0], cmap='binary')
plt.axis('off')
plt.show()

# %%
y_train

# %%
X_valid.shape, X_valid.dtype

# %%
X_test.shape, X_test.dtype
# %%
# Let's plot images in the dataset
n_cols = 10
n_rows = 4
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))

for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)

        plt.imshow(X_train[index], cmap='binary')
        plt.axis('off')
        plt.title(y_train[index], fontsize=12)

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# %%


class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, epoch, logs={}):
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])

        newRate = self.model.optimizer.learning_rate * self.factor

        keras.backend.set_value(
            self.model.optimizer.learning_rate,
            newRate
        )


# %%
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# %%
# Restrict the memory consumption
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    except RuntimeError as e:
        print(e)

# %%
# Create a Sequential model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# %%
# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.SGD(learning_rate=1e-3),
    metrics=['accuracy']
)
exp_lr = ExponentialLearningRate(factor=1.005)

# %%
# Fit the model
model.fit(
    X_train, y_train,
    epochs=1,
    validation_data=(X_valid, y_valid),
    callbacks=[exp_lr]
)

# %%
plt.plot(exp_lr.rates, exp_lr.losses)
plt.gca().set_xscale('log')

plt.hlines(
    min(exp_lr.losses),
    min(exp_lr.rates),
    max(exp_lr.rates)
)

plt.vlines(6e-1, 0, ymax=max(exp_lr.losses))

plt.axis([
    min(exp_lr.rates), max(exp_lr.rates),
    0, exp_lr.losses[0]
])

plt.grid()

plt.xlabel("Learning rate")
plt.ylabel("Loss")

# %% [markdown]
# > Use half of the learning rate for the first epoch = 6e-1 / 2 = 3e-1
# %%
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# %%
# Create a Sequential model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# %%
# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.SGD(learning_rate=3e-1),
    metrics=['accuracy']
)

# %%
# Define root log directory

root_logdir = os.path.join(os.curdir, "logs_mnist")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()

# %%
# Early stopping callback
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=20
)
# Checkpoint callback
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'my_mnist_model.h5',
    save_best_only=True
)
# TensorBoard callback
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# %%
# Fit the model
model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_valid, y_valid),
    callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb]
)
# %%
# Load the best model
model = keras.models.load_model('my_mnist_model.h5')
# Accuracy on the test set
model.evaluate(X_test, y_test)
# %%
# Look at the learning curves on the tensorboard
