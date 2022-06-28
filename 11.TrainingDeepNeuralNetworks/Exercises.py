# %%
import os
from IPython import get_ipython

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
# %%
tf.config.experimental.list_physical_devices('GPU')
# %% [markdown]
# #### The CIFAR-10 dataset
# (Canadian Institute For Advanced Research) is a collection
# of images that are commonly used to train machine learning
# and computer vision algorithms.
# It is one of the most widely used datasets
# for machine learning research.[1][2] The CIFAR-10
# dataset contains 60,000 32x32 color images in 10
# different classes.[3] The 10 different classes represent
# airplanes, cars, birds, cats, deer, dogs, frogs, horses,
# ships, and trucks. There are 6,000 images of each class
# %%
# Load the CIFAR10 dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_train = X_train_full[5000:]
y_train = y_train_full[5000:]
X_valid = X_train_full[:5000]
y_valid = y_train_full[:5000]
# %%
X_train.shape
# %%
# Let's take a look at the first images in the training set
plt.figure(figsize=(9, 9))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_train[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
# %%
# Check y_train data
y_train[:25]
# %%
# Create a model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))

for _ in range(20):
    model.add(keras.layers.Dense(
        100,
        activation='elu',
        kernel_initializer='he_normal'
    ))
model.add(keras.layers.Dense(10, activation='softmax'))
# %%
# Create Nadam optimizer
optimizer = keras.optimizers.Nadam(learning_rate=1e-4)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)
# %%
# Tensorboard callback
run_index = 1  # increment every time you train the model
run_logdir = os.path.join(os.curdir, "my_cifar10_logs",
                          "run_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# Crate early stopping callback
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=10
)
# Model checkpoint callback
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'my_cifar10_model.h5',
    save_best_only=True,
    verbose=1
)
callbacks = [tensorboard_cb, early_stopping_cb, checkpoint_cb]
# %%
model.fit(X_train, y_train, epochs=100,
          validation_data=(X_valid, y_valid),
          callbacks=callbacks)
# %%
model.evaluate(X_valid, y_valid)
# %%
# Check the best model
model = keras.models.load_model('my_cifar10_model.h5')
model.evaluate(X_valid, y_valid)
# %%
# Launch Tensorboard session
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().system('tensorboard --logdir ./my_cifar10_logs --port 6006')

# %%
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# %%
# Create a model with BatchNormalization
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
model.add(keras.layers.BatchNormalization())

for _ in range(20):
    model.add(keras.layers.Dense(
        100,
        kernel_initializer='he_normal'
    ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('elu'))

model.add(keras.layers.Dense(10, activation='softmax'))

# %%
# Create Nadam optimizer and compile the model
optimizer = keras.optimizers.Nadam(learning_rate=1e-4)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# %%
# Tensorboard callback
run_index = 1  # increment every time you train the model
run_logdir = os.path.join(os.curdir, "my_cifar10_logs",
                          "run_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# Crate early stopping callback
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=10
)
# Model checkpoint callback
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'my_cifar10_bn_model.h5',
    save_best_only=True,
    verbose=1
)
callbacks = [tensorboard_cb, early_stopping_cb, checkpoint_cb]
# %%
model.fit(X_train, y_train, epochs=100,
          validation_data=(X_valid, y_valid),
          callbacks=callbacks)
# %%
model = keras.models.load_model('my_cifar10_bn_model.h5')
model.evaluate(X_valid, y_valid)

# %%
