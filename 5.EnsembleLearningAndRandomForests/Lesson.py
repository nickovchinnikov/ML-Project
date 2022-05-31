# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import (
    make_moons,
    load_iris,
    fetch_openml
)
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingRegressor
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error
)

import xgboost as xgb

# %%
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# %%
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# %% [markdown]
# ### Bagging and Pasting
# %%
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42),
    n_estimators=500,
    bootstrap=True,
    n_jobs=-1,
)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

accuracy_score(y_test, y_pred)

# %%
# Out-of-Bag Evaluation
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42),
    n_estimators=500,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_

# %%
y_pred = bag_clf.predict(X_test)

accuracy_score(y_test, y_pred)

# %%
bag_clf.oob_decision_function_[:500:50]

# %% [markdown]
# ### Random Forests 
# %%
rnd_clf = RandomForestClassifier(
    n_estimators=500,
    max_leaf_nodes=16,
    n_jobs=-1,
    random_state=42
)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)

accuracy_score(y_test, y_pred_rf)

# %%
# Roughly the same accuracy as the bagging classifier
# with the following parameters:
bag_clf2 = BaggingClassifier(
    DecisionTreeClassifier(
        splitter="random",
        max_leaf_nodes=16
    ),
    n_estimators=500,
    max_samples=1.0,
    bootstrap=True,
    n_jobs=-1
)
bag_clf2.fit(X_train, y_train)

y_pred_bc2 = bag_clf2.predict(X_test)

accuracy_score(y_test, y_pred_bc2)

# %% [markdown]
# ### Extra-Trees
# %%
extra_clf = ExtraTreesClassifier(
    n_estimators=500,
    max_leaf_nodes=16,
    n_jobs=-1,
    random_state=42
)
extra_clf.fit(X_train, y_train)

y_pred_extra = extra_clf.predict(X_test)

accuracy_score(y_test, y_pred_extra)

# %% [markdown]
# ### Feature Importance
# %%
iris = load_iris()
rnd_clf = RandomForestClassifier(
    n_estimators=500,
    n_jobs=-1,
    random_state=42
)
rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

# %%
mnist = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
mnist.target = mnist.target.astype(np.uint8)

rnd_clf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    random_state=42
)
rnd_clf.fit(mnist["data"], mnist["target"])

# %%
def plot_digits(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=plt.cm.hot, interpolation="nearest")
    plt.axis("off")

# %%
plot_digits(rnd_clf.feature_importances_)
cbar = plt.colorbar(
    ticks=[
        rnd_clf.feature_importances_.min(),
        rnd_clf.feature_importances_.max()
    ]
)
cbar.ax.set_yticklabels(["Not important", "Very important"])

plt.show()

# %% [markdown]
# ### AdaBoost Classifier
# %%
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    algorithm="SAMME.R",
    learning_rate=0.5,
)

ada_clf.fit(X_train, y_train)

y_pred_ada = ada_clf.predict(X_test)

# %% [markdown]
# ### Gradient Boosting
# %%
tree_reg_1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg_1.fit(X_train, y_train)

y_pred1 = tree_reg_1.predict(X_train)

mse_1 = mean_squared_error(y_train, y_pred1)
rmse_1 = np.sqrt(mse_1)
rmse_1

# %%
# Traint the next predictor on the residual errors of the previous predictor
tree_reg_2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg_2.fit(X_train, y_train - y_pred1)

y_pred2 = tree_reg_2.predict(X_train)

mse_2 = mean_squared_error(y_train, y_pred2)
rmse_2 = np.sqrt(mse_2)
rmse_2

# %%
tree_reg_3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg_3.fit(X_train, y_train - y_pred2)

y_pred3 = tree_reg_3.predict(X_train)

mse_3 = mean_squared_error(y_train, y_pred3)
rmse_3 = np.sqrt(mse_3)
rmse_3

# %%
# Adding up the predictions of the three predictors
y_pred = sum(
    tree.predict(X_test) for tree in (
        tree_reg_1,
        tree_reg_2,
        tree_reg_3
    )
)

mse_3 = mean_squared_error(y_test, y_pred)
rmse_3 = np.sqrt(mse_3)
rmse_3

# %% [markdown]
# ### Gradient Boosting Regression
# %%
gbrt = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=3,
    learning_rate=1
)

gbrt.fit(X_train, y_train)

y_pred = gbrt.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
np.sqrt(mse)

# %% [markdown]
# ### Early stopping
# %%
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y)

gbrt = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=120
)

gbrt.fit(X_train, y_train)

errors = [
    mean_squared_error(y_test, y_pred)
    for y_pred in gbrt.staged_predict(X_test)
]

bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=bst_n_estimators
)

gbrt_best.fit(X_train, y_train)

# %%
# Generate tunning curve
min_error = np.min(errors)
min_error

# %%
def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)
# %%
plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.plot(errors, "b.-")
plt.plot([0, 120], [min_error, min_error], "k--")
plt.plot(bst_n_estimators, min_error, "ko")
plt.plot([bst_n_estimators, bst_n_estimators], [min_error, 0], "k--")
plt.xlabel("# of trees")
plt.ylabel("MSE")
plt.title("Validation error")
plt.axis([0, 120, 0, 0.01])

plt.subplot(122)
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.xlabel("$x_1$", fontsize=16)

plt.show()
# %%
# Early stop with warm start
gbrt = GradientBoostingRegressor(
    max_depth=2,
    warm_start=True
)

max_error = float("inf")
error_going_up = 0

for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    if mse < max_error:
        max_error = mse
        gbrt.n_estimators = n_estimators
    else:
        error_going_up += 1
        if error_going_up == 5:
            break

print ("Best number of estimators:", gbrt.n_estimators)

# %% [markdown]
# ### XGB Boosting
# %%
xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(X_train, y_train)

y_pred = xgb_reg.predict(X_test)
# %%
# Early stop
xgb_reg.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=5,
    eval_metric="rmse"
)