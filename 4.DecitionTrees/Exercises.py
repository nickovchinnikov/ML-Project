# %%
import time

import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    ShuffleSplit
)
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from scipy.stats import mode

# %%
data = make_moons(n_samples=10000, noise=0.4, random_state=42)
X, y = data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train[:5]
# %%
# Ex. 7: Train and fine-tune a Decision Tree for the moons dataset.
# Find the best parameters for the decision tree classifier using a
# grid search with cross validation

# With max depth = 9 and max_leaf_nodes = 48 we get a score of 0.8605
# max_depth = list(range(3, 14))
max_leaf_nodes = list(range(2, 100))
min_samples_split = [2, 3, 4]

params_grid = {
    # 'max_depth': max_depth,
    'max_leaf_nodes': max_leaf_nodes,
    'min_samples_split': min_samples_split,
}

start_time = time.perf_counter()

grid_search_cv = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    params_grid,
    cv=3,
    verbose=1
)

grid_search_cv.fit(X_train, y_train)

end_time = time.perf_counter()

print(end_time - start_time, "seconds")

# %%
# The best estimator is:
grid_search_cv.best_estimator_

# %%
# The accuracy score is:
y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
# Ex. 8: Grow a forest.
# Train a decision tree classifier on the moons dataset and grow a forest of trees.
n_trees = 1000
n_instances = 100

mini_sets = []

len(X_train) - n_instances

rs = ShuffleSplit(
    n_splits=n_trees,
    test_size=len(X_train) - n_instances,
    random_state=42
)

for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

# %%
# Create a random forest classifier
forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)

    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

np.mean(accuracy_scores)

# %%
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

# %%
# Majority vote prediction of the forest:
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
y_pred_majority = y_pred_majority_votes.reshape(-1)
accuracy_score(y_test, y_pred_majority)
