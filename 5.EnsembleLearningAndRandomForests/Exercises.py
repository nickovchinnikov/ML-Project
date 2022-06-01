# %%
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# %% [markdown]
# ### Exercise 8:
# %%
# Download the mnist dataset from OpenML:
mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8)

# %%
# Split the data into training and test sets:
data, target = mnist["data"], mnist["target"]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    data,
    target,
    test_size=10000,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=10000,
    random_state=42
)

len(X_test), len(X_val), len(X_train)

# %%
# Create a set of classifiers:
clfs = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    ExtraTreesClassifier(n_estimators=100, random_state=42),
    LinearSVC(max_iter=100, tol=20, random_state=42),
    MLPClassifier(random_state=42),
]

for clf in clfs:
    print("Training the", clf)
    clf.fit(X_train, y_train)
    print("Score:", clf.score(X_val, y_val))

# %%
# Create a voting classifier:
named_clfs = [
    ("random_forest", clfs[0]),
    ("extra_trees", clfs[1]),
    ("linear_svc", clfs[2]),
    ("mlp", clfs[3]),
]
voting_clf = VotingClassifier(named_clfs)

# %%
voting_clf.fit(X_train, y_train)

voting_clf.score(X_val, y_val)

# %%
[estimator.score(X_val, y_val) for estimator in voting_clf.estimators_]
# %%
# Drop the LinearSVC classifier:
voting_clf.set_params(linear_svc=None)

# %%
voting_clf.estimators

# %%
voting_clf.estimators_

# %%
del voting_clf.estimators_[2]

# %%
voting_clf.score(X_val, y_val)

# %%
# XGBoost classifier:
xgb_clf = XGBClassifier(use_label_encoder=False, random_state=42)
xgb_clf.fit(X_train, y_train)

xgb_clf.score(X_val, y_val)

# %% [markdown]
# Score of XGBClassifier is higher than VotingClassifier!
# %%
