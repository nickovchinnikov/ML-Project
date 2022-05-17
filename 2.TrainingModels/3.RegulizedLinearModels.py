# %%
from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from sklearn import clone, datasets
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    SGDRegressor,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# %% [markdown]
"""
# Regularized Linear Models

## Ridge Regression

#### Equation 4-8. Ridge Regression cost function
    \
$J(θ) = MSE(θ) + \alpha \frac{1}{2} \sum_{i=1}^n θ_i^2$
    \
    \
Note that the bias term $θ_0$ is not regularized (the sum starts at $i = 1$, not 0).
If we define w as the vector of feature weights ($θ_1$ to $θ_n$), then the regularization
term is simply equal to $\frac{1}{2}(\| w \|_2)^2$, where $\|w\|^2$ represents the $\ell_2$
norm of the weight vector. For Gradient Descent, just add $\alpha{w}$ to the MSE
gradient vector

> It is common to use the notation $J(θ)$ for cost functions that don’t have a short name;
we will often use this notation throughout the rest of this book. The context will make
it clear which cost function is being discussed

As with Linear Regression, we can perform Ridge Regression either by computing a
closed-form equation or by performing Gradient Descent. The pros and cons are the
same. Equation 4-9 shows the closed-form solution (where $A$ is the $(n + 1) \times (n + 1)$
identity matrix13 except with a 0 in the top-left cell, corresponding to the bias term).

#### Equation 4-9. Ridge Regression closed-form solution
$\theta = (X^T X + \alpha{A})^{−1} X^T y$
    \
Here is how to perform Ridge Regression with Scikit-Learn using a closed-form
solution (a variant of Equation 4-9 using a matrix factorization technique by André-Louis
Cholesky):
"""
# %%
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Split the data into training/testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# %%
linear_reg = LinearRegression()
linear_reg.fit(X, y)

linear_reg.predict([[1.5]])

# %%
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)

# %%
ridge_reg.predict([[1.5]])

# %% [markdown]
"""
And using Stochastic Gradient Descent
"""
# %%
sgd_reg = SGDRegressor(penalty='l2')
sgd_reg.fit(X, np.ravel(y))
sgd_reg.predict([[1.5]])

# %% [markdown]
"""
The penalty hyperparameter sets the type of regularization term to use. Specifying
"l2" indicates that you want SGD to add a regularization term to the cost function
equal to half the square of the $\ell_2$ norm of the weight vector: this is simply Ridge
Regression.

## Lasso Regression

Least Absolute Shrinkage and Selection Operator Regression (simply called Lasso
Regression) is another regularized version of Linear Regression: just like Ridge
Regression, it adds a regularization term to the cost function, but it uses the $\ell_1$
norm of the weight vector instead of half the square of the $\ell_2$ norm (see Equation 4-10).

#### Equation 4-10. Lasso Regression cost function

$J(\theta) = MSE(θ) + \alpha\sum_{i=1}^n|θ_i|$
    \
The Lasso cost function is not differentiable at $θ_i = 0$ (for $i = 1, 2, \dots , n$), 
but Gradient Descent still works fine if you use a *subgradient* vector $g$ instead
when any $θ_i = 0$. Equation 4-11 shows a subgradient vector equation you can use for
Gradient Descent with the Lasso cost function.

#### Equation 4-11. Lasso Regression subgradient vector

$g(\theta, J)= \nabla_{\theta} MSE(\theta) + \alpha \begin{pmatrix}
    sign(\theta_1) \\
    sign(\theta_2) \\
    \vdots \\
    sign(\theta_n)
\end{pmatrix}$

Where

$sign(\theta_1) = \begin{cases}
    -1  & \quad \text{if } \theta_i < 0 \\
     0  & \quad \text{if } \theta_i = 0 \\
    +1  & \quad \text{if } \theta_i > 0
  \end{cases}$
    \
Here is a small Scikit-Learn example using the Lasso class. Note that you could
instead use an `SGDRegressor(penalty="l1")`
"""
# %%
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)

lasso_reg.predict([[1.5]])

# %% [markdown]
"""
## Elastic Net

Elastic Net is a middle ground between Ridge Regression and Lasso Regression. The
regularization term is a simple mix of both Ridge and Lasso’s regularization terms,
and you can control the mix ratio r. When $r = 0$, Elastic Net is equivalent to Ridge
Regression, and when $r = 1$, it is equivalent to Lasso Regression (see Equation 4-12).

#### Equation 4-12. Elastic Net cost function

$J(θ) = MSE(θ) + r \alpha \sum_{i=1}^n |θ_i| + \frac{1−r}{2} \alpha \sum_{i=1}^n θ_i^2$
    \
    \
Here is a short example using Scikit-Learn’s ElasticNet (l1_ratio corresponds to
the mix ratio r)
"""
# %%
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])

# %% [markdown]
"""
## Early Stopping
"""
# %%
# prepare the data
poly_scaler = Pipeline([
    ('poly_features', PolynomialFeatures(degree=90, include_bias=False)),
    ('std_scaler', StandardScaler())
])
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(
    max_iter=1,
    tol=-np.infty,
    warm_start=True,
    penalty=None,
    learning_rate="constant",
    eta0=0.0005
)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train) # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)

# %% [markdown]
"""
# Logistic Regression

## Estimating Probabilities

#### Equation 4-13. Logistic Regression model estimated probability (vectorized form)

$\hat{p} = h_{\theta}(x) = \sigma(x^T \theta)$
\
$\sigma(\cdot)$ — is a sigmoid function (i.e., S-shaped) that outputs a number
between 0 and 1. It is defined as shown in Equation 4-14

#### Equation 4-14. Logistic function

$\sigma(t)=\frac{1}{1 + exp(-t)}$

"""
# %%
t = np.linspace(-10, 10, 100)

def sigma(t):
    return 1 / (1 + np.exp(-t))

plt.figure(figsize=(9, 3))
plt.plot(t, sigma(t), 'b-', linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")

plt.plot([-10, 10], [0, 0], 'k-')
plt.plot([-10, 10], [0.5, 0.5], 'k:')
plt.plot([-10, 10], [1, 1], 'k:')
plt.plot([0, 0], [-1.1, 1.1], 'k-')

plt.axis([-10, 10, -0.1, 1.1])
plt.legend(loc="upper left", fontsize=20)

# %% [markdown]
"""
Once the Logistic Regression model has estimated the probability $p = h_θ(x)$ that an
instance $x$ belongs to the positive class, it can make its prediction $\hat{y}$ easily

#### Equation 4-15. Logistic Regression model prediction

$\hat{y} = \begin{cases}
    0  & \quad \text{if } \hat{p} < 0.5 \\
    1  & \quad \text{if } \hat{p} \geq 0.5
  \end{cases}$
\
Notice that $\sigma(t) < 0.5$ when $t < 0$, and $\sigma(t) \geq 0.5$ when $t \geq 0$,
so a Logistic Regression model predicts 1 if $x^T\theta$ is positive, and 0 if it is negative.

## Decision Boundaries
"""
# %%
iris = datasets.load_iris()
list(iris.keys())

# %%
# Petal width
X = iris["data"][:, 3:]

# 1 if Iris-Virginica, else 0
y = (iris["target"] == 2).astype(np.int32)

# Print the first 5 rows of X
X[:5], y[:5]
# %%

log_reg = LogisticRegression()
log_reg.fit(X, y)

# %% [markdown]
"""
Let’s look at the model’s estimated probabilities for flowers with petal widths varying
from 0 to 3 cm
"""
# %%
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")

# %% [markdown]
"""
The petal width of Iris-Virginica flowers (represented by triangles) ranges from 1.4
cm to 2.5 cm, while the other iris flowers (represented by squares) generally have a
smaller petal width, ranging from 0.1 cm to 1.8 cm. Notice that there is a bit of over‐
lap. Above about 2 cm the classifier is highly confident that the flower is an Iris-
Virginica (it outputs a high probability to that class), while below 1 cm it is highly
confident that it is not an Iris-Virginica (high probability for the “Not Iris-Virginica”
class). In between these extremes, the classifier is unsure. However, if you ask it to
predict the class (using the `predict()` method rather than the `predict_proba()`
method), it will return whichever class is the most likely. Therefore, there is a decision
boundary at around 1.6 cm where both probabilities are equal to 50%: if the petal
width is higher than 1.6 cm, the classifier will predict that the flower is an Iris-
Virginica, or else it will predict that it is not (even if it is not very confident):
"""

# %%
log_reg.predict([[1.7], [1.5]])

# %% [markdown]
"""
# Softmax Regression
"""
# %%
X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X, y)
# %%
softmax_reg.predict([[5, 2]])

# %%
softmax_reg.predict_proba([[5, 2]])
