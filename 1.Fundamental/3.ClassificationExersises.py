# %%
import time
from typing import Literal, cast, List

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_predict
from scipy.ndimage.interpolation import shift

# %% [markdown]
"""
# Exercises

1. Try to build a classifier for the MNIST dataset that achieves over 97% accuracy
on the test set. Hint: the `KNeighborsClassifier` works quite well for this task;
you just need to find good hyperparameter values (try a grid search on the
`weights` and `n_neighbors` hyperparameters).
"""
# %%
mnist = fetch_openml("mnist_784", version=1)
mnist.keys()
# %%
X, y = mnist.data, mnist.target

X.shape, y.shape
# %%
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

some_digit = X[0]
# %%
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
# %%
knn_clf.predict([some_digit])
# %%
y_train_knn_predict = cross_val_predict(
    knn_clf, X_train, y_train, cv=3, n_jobs=3)
# %%
f1_score(y_train, y_train_knn_predict, average="macro")
# %% [markdown]
"""
Very good score 96.722% !

# Try `GridSearchCV` optimization
"""
# %%
start_time = time.perf_counter()

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

knn_clf = KNeighborsClassifier()

grid_search = GridSearchCV(
    knn_clf,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    verbose=3
)
grid_search.fit(X_train, y_train)

end_time = time.perf_counter()

print(end_time - start_time, "seconds")
# %% [markdown]
"""
# Let's take a look what are the best options that we have and score
"""
# %%
grid_search.best_params_
# %%
grid_search.best_score_
# %% [markdown]
"""
2. Write a function that can shift an MNIST image in any direction (left, right, up,
or down) by one pixel. Then, for each image in the training set, create four shifted
copies (one per direction) and add them to the training set. Finally, train your
best model on this expanded training set and measure its accuracy on the test set.
You should observe that your model performs even better now! This technique of
artificially growing the training set is called data augmentation or training set
expansion.
"""
# %%


def shift_image(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx])
    return shifted_image.reshape(-1)


# %%
image = X_train[600]
shifted_image_down = shift_image(image, 0, 5)
shifted_image_left = shift_image(image, -5, 0)

plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title("Original", fontsize=14)
plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(132)
plt.title("Shifted down", fontsize=14)
plt.imshow(shifted_image_down.reshape(28, 28),
           interpolation="nearest", cmap="Greys")
plt.subplot(133)
plt.title("Shifted left", fontsize=14)
plt.imshow(shifted_image_left.reshape(28, 28),
           interpolation="nearest", cmap="Greys")
plt.show()

# %%
X_train_a = [image for image in X_train]
y_train_a = [label for label in y_train]

print(len(X_train_a), len(y_train_a))

for image, label in zip(X_train, y_train):
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted_image = shift_image(image, dy, dx)
        X_train_a.append(shifted_image)
        y_train_a.append(label)

X_train_augmented = np.array(X_train_a)
y_train_augmented = np.array(y_train_a)

print(len(X_train_augmented), len(y_train_augmented))
# %%
shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]
# %%
knn_clf = KNeighborsClassifier(n_neighbors=4, weights="distance")
# %%
knn_clf.fit(X_train_augmented, y_train_augmented)
# %%
knn_clf.predict([some_digit])
# %%
y_pred = knn_clf.predict(X_test)
accuracy_score(y_test, y_pred)
