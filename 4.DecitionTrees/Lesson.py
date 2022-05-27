# %%
from sklearn.datasets import load_iris

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree
)

# %%
# Load data
iris = load_iris()
print(iris.DESCR)

X = iris.data[:, 2:]  # petal length and width
y = iris.target

X[:5], y[:5]

# %%
# Decision tree classifier
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

# %%
# Visualize tree
plot_tree(tree_clf)

# %%
# Predict flower type probability
tree_clf.predict_proba([[5, 1.5]])

# %%
# Predict flower type class
tree_clf.predict([[5, 1.5]]), tree_clf.predict([[2.4, 1.5]])

# %%
# Decision tree regressor
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)

# %%
# Visualize tree
plot_tree(tree_reg)

# %%
