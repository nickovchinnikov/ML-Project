# %%
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import reciprocal, uniform

from sklearn import datasets
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# %% [markdown]
# Exercise 8: train a `LinearSVC` on a linearly separable dataset.
# Then train an `SVC` and a `SGDClassifier` on the same dataset.
# See if you can get them to produce roughly the same model.

# %%
# Load the iris dataset
iris = datasets.load_iris()

X = iris["data"][:, [2, 3]] # petal length, petal width
y = iris["target"]

# Filter the dataset to get only 2-class data
setosa_or_versicolor = (y == 0) | (y == 1)

X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# Print for debug first five rows of X and y
X[:5], y[:5]

# %%
m = len(X)
C = 5
alpha = 1 / (C * m)

# Prepare the models
lin_clf = LinearSVC(loss="hinge", C=C, random_state=42)

svm_clf = SVC(kernel="linear", C=C)

sgd_clf = SGDClassifier(
    loss="hinge",
    learning_rate="constant",
    eta0=0.001,
    alpha=alpha,
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

# Prepare the data for the models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lin_clf.fit(X_scaled, y)
svm_clf.fit(X_scaled, y)
sgd_clf.fit(X_scaled, y)

# Print the results

print("LinearSVC:", lin_clf.score(X_scaled, y))
print("LinearSVC interception and coff:", lin_clf.intercept_, lin_clf.coef_)

print("SVC:", svm_clf.score(X, y))
print("SVC interception and coff:", svm_clf.intercept_, svm_clf.coef_)

print("SGDClassifier:", sgd_clf.score(X, y))
print("SGDClassifier(alpha={:.5f}) intercept and coff:".format(sgd_clf.alpha),
    sgd_clf.intercept_, sgd_clf.coef_)

# %% [markdown]
# Let's plot the decision boundary of the models
# Create line for the models
def get_line_from_clf(clf, X, y):
    s = -clf.coef_[0, 0] / clf.coef_[0, 1]
    b = -clf.intercept_[0] / clf.coef_[0, 1]
    line = scaler.inverse_transform([[-2, -2 * s + b], [2, 2 * s + b]])
    return line

# %%
line1 = get_line_from_clf(lin_clf, X, y)
line2 = get_line_from_clf(svm_clf, X, y)
line3 = get_line_from_clf(sgd_clf, X, y)

plt.figure(figsize=(11, 4))

# Plot all three decision boundaries
plt.plot(line1[:, 0], line1[:, 1], "k:", label="LinearSVC")
plt.plot(line2[:, 0], line2[:, 1], "b--", label="SVC")
plt.plot(line3[:, 0], line3[:, 1], "r-", label="SGDClassifier")

# Plot the training data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)

plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.legend(loc="upper center", fontsize=14)

plt.axis([0, 5.5, 0, 2])
plt.show()

# %% [markdown]
# Very close to each other!
# %% [markdown]
# _Exercise 9: train an SVM classifier on the MNIST dataset.
# Since SVM classifiers are binary classifiers, you will need to use
# one-versus-all to classify all 10 digits. You may want to tune the
# hyperparameters using small validation sets to speed up the process.
# What accuracy can you reach?_

# %%
mnist = datasets.fetch_openml("mnist_784", version=1, cache=True, as_frame=False)

# %%
X = mnist["data"]
y = mnist["target"]

X_train = X[:60000]
y_train = y[:60000]

X_test = X[60000:]
y_test = y[60000:]

# %%
# Train the models
lin_clf = LinearSVC(random_state=42)
lin_clf.fit(X_train, y_train)

# %%
# Measure the accuracy of the models
y_pred = lin_clf.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
# Let's rescale the data to make better training

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

# %%
# Train the models again
lin_clf = LinearSVC(random_state=42)
lin_clf.fit(X_train_scaled, y_train)

# %%
# Measure the accuracy of the models
y_pred = lin_clf.predict(X_test_scaled)
accuracy_score(y_test, y_pred)

# %%
# Let's train SVC classifier on the MNIST dataset
svm_clf = SVC(gamma="scale")
svm_clf.fit(X_train_scaled[:10000], y_train[:10000])

# %%
# Measure the accuracy of the models
y_pred = svm_clf.predict(X_train_scaled)
accuracy_score(y_train, y_pred)

# %%
# Randomized search for hyperparameters
param_distributions = {
    "gamma": reciprocal(0.001, 0.1),
    "C": uniform(1, 10)
}

rnd_search_cv = RandomizedSearchCV(
    svm_clf,
    param_distributions,
    n_iter=10,
    verbose=2,
    cv=3
)

rnd_search_cv.fit(X_train_scaled[:1000], y_train[:1000])

# %%
# The best estimator params
rnd_search_cv.best_estimator_

# %%
# The best estimator score
rnd_search_cv.best_score_

# %%
# Fit the best estimator on the whole training set
rnd_search_cv.best_estimator_.fit(X_train_scaled, y_train)

# %%
# Measure the accuracy of the models for the training set
y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
accuracy_score(y_train, y_pred)

# %%
# Measure the accuracy of the models for the test set
y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
accuracy_score(y_test, y_pred)

# %% [markdown]
# _Exercise: train an SVM regressor on the California housing dataset._

# %%
# Create model with the best C and gamma parameters
svm_clf = SVC(C=5, gamma=0.005)
svm_clf.fit(X_train_scaled, y_train)

# %%
# Measure the accuracy of the models for the training set
y_pred = svm_clf.predict(X_train_scaled)
accuracy_score(y_train, y_pred)

# %%
# Measure the accuracy of the models for the test set
y_pred = svm_clf.predict(X_test_scaled)
accuracy_score(y_test, y_pred)

# %% [markdown]
# _Exercise 10: train an SVM regressor on the California housing dataset._

# %%
housing = datasets.fetch_california_housing()

X = housing["data"]
y = housing["target"]

# %%
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train[:5]

# %%
# Scale the data to make better training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Let's train LinearSVR classifier on the scaled data
lin_svr = LinearSVR(random_state=42)
lin_svr.fit(X_train_scaled, y_train)

# %%
# Measure the accuracy of the models for the training set by MSE
y_pred = lin_svr.predict(X_train_scaled)
mse = mean_squared_error(y_train, y_pred)
mse

# %%
# RMSE
rmse = np.sqrt(mse)
rmse

# %%
# Not so good! Let's use randomized search to find the best hyperparameters
param_distributions = {
    "gamma": reciprocal(0.001, 0.1),
    "C": uniform(1, 10)
}

rnd_search_cv = RandomizedSearchCV(
    SVR(),
    param_distributions,
    n_iter=10,
    verbose=2,
    cv=3,
    random_state=42
)
rnd_search_cv.fit(X_train_scaled, y_train)

# %%
# The best estimator params
rnd_search_cv.best_estimator_

# %%
# The best estimator score
rnd_search_cv.best_score_

# %%
# The best estimator prediction on the training set
# Let's measure RMSE on the training set
y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
rmse

# %%
# The best estimator prediction on the test set
# Let's measure RMSE on the test set
y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse

# %%
