# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
# Split data to train and test subsets
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
# %% [markdown]
"""
### Let’s generate some linear-looking data to test this equation on
"""
# %%
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# %%
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
X_b[:5]
# %% [markdown]
"""
### Equation 4-6. Gradient vector of the cost function

$\nabla_{\theta}\text{MSE}(\theta) = \frac{2}{m}X^T(X\theta - y)$

### Equation 4-7. Gradient Descent step

$\theta^{\text{(next step)}} = \theta − \eta \nabla_\theta MSE(\theta)$

[Proof for Gradient Descent step](https://math.stackexchange.com/questions/3152743/proof-of-batch-gradient-descents-cost-function-gradient-vector/4379530#4379530)

"""
# %% [markdown]
"""
Let’s look at a quick implementation of this algorithm
"""
# %%
# Learning rate
eta = 0.1
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1)

for iteration in range(n_iterations):
    gradient = (2/m) * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradient
theta

# %% [markdown]
"""
This code implements Stochastic Gradient Descent using a simple learning schedule
"""
# %%
# Learning schedule hyperparameters
n_epochs = 50
t0, t1 = 5, 50
m = 100

def learning_schedule(t):
    return t0 / (t + t1)

# random init
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

theta
# %%
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
sgd_reg.intercept_, sgd_reg.coef_
# %% [markdown]
"""
## Polynomial Regression
"""
# %%
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Plot the result of generation
plt.plot(X, y, 'b.')
# %% [markdown]
"""
Clearly, a straight line will never fit this data properly.
So let’s use Scikit-Learn’s `PolynomialFeatures` class to
transform our training data, adding the square (2nd-degree
polynomial) of each feature in the training set as new features
(in this case there is just one feature):
"""
# %%
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0], X_poly[0]
# %% [markdown]
"""
`X_poly` now contains the original feature of `X` plus the square
of this feature. Now you can fit a `LinearRegression` model to
this extended training data
"""
# %%
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
# %%
coef = lin_reg.coef_[0]

# Restore the function with calculated coeff
# y_predicted = coef[1] * (X**2) + coef[0] * X + lin_reg.intercept_

# The same result as above
y_predicted = lin_reg.predict(X_poly)

fig = plt.figure(figsize=(8, 2.5), facecolor="#f1f1f1")
ax = fig.add_axes((1, 1, 1, 1))

ax.plot(X, y, 'b.')
ax.plot(X, y_predicted, 'ro')

# %% [markdown]
"""
Another way is to look at the learning curves: these are
plots of the model’s performance on the training set and
the validation set as a function of the training set size
(or the training iteration). To generate the plots, simply
train the model several times on different sized subsets of
the training set. The following code defines a function that
plots the learning curves of a model given some training data:
"""
# %%
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_error, val_error = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_error.append(mean_squared_error(y_train[:m], y_train_predict))
        val_error.append(mean_squared_error(y_val, y_val_predict))
    fig = plt.figure(figsize=(10, 5), facecolor="#f1f1f1")
    ax = fig.add_axes((1, 1, 1, 1))
    ax.set_xlim([0, 80])
    ax.set_ylim([0, 5])
    plt.plot(np.sqrt(train_error), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_error), "b-", linewidth=3, label="val")
# %%
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
# %% [markdown]
"""
This deserves a bit of explanation. First, let’s look at the performance on the training
data: when there are just one or two instances in the training set, the model can fit
them perfectly, which is why the curve starts at zero. But as new instances are added
to the training set, it becomes impossible for the model to fit the training data per‐
fectly, both because the data is noisy and because it is not linear at all. So the error on
the training data goes up until it reaches a plateau, at which point adding new instan‐
ces to the training set doesn’t make the average error much better or worse. Now let’s
look at the performance of the model on the validation data. When the model is
trained on very few training instances, it is incapable of generalizing properly, which
is why the validation error is initially quite big. Then as the model is shown more
training examples, it learns and thus the validation error slowly goes down. However,
once again a straight line cannot do a good job modeling the data, so the error ends
up at a plateau, very close to the other curve.
These learning curves are typical of an underfitting model. Both curves have reached
a plateau; they are close and fairly high.

> If your model is underfitting the training data, adding more training
examples will not help. You need to use a more complex model
or come up with better features.

### Now let’s look at the learning curves of a 10th-degree polynomial model on the same data
"""
# %%
polynomal_regression = Pipeline([
    ('poly_features', PolynomialFeatures(degree=10, include_bias=False)),
    ('lin_regression', LinearRegression())
])
plot_learning_curves(polynomal_regression, X, y)

# %%
