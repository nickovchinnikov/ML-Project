# %%
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as ln

from sklearn.linear_model import LinearRegression

# %% [markdown]
"""
# Linear regression
"""
# %% [markdown]
"""
### Let’s generate some linear-looking data to test this equation on
"""
# %%
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# %%
plt.plot(X, y, 'b.')
# %% [markdown]
"""
> Randomly generated linear dataset
"""
# %% [markdown]
"""
Now let’s compute $\hat{\theta}$ using the Normal Equation. We will use 
the `inv()` function from NumPy’s Linear Algebra module (`np.linalg`) to 
compute the inverse of a matrix, and the `dot()` method for matrix multiplication:
"""
# %%
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
X_b[:5]
# %%
theta_best = ln.inv(X_b.T @ X_b) @ X_b.T @ y
# %% [markdown]
"""
The actual function that we used to generate the data is
$y = 4 + 3X_1 + \textrm{Gaussian noise}$.
Let’s see what the equation found:
"""
# %%
theta_best
# %% [markdown]
"""
Now you can make predictions using $\hat{\theta}$:
"""
# %%
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
X_new_b
# %%
y_predict = X_new_b @ theta_best
y_predict
# %% [markdown]
"""
Let’s plot this model’s predictions:
"""
# %%
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, ".b")
# %% [markdown]
"""
> Linear Regression model predictions
"""
# %%
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
# %%
lin_reg.predict(X_new)
# %% [markdown]
"""
The `LinearRegression` class is based on the `np.linalg.lstsq()`
function (the name stands for “least squares”), which you could
call directly:
"""
# %%
theta_best_svd, residuals, rank, s = ln.lstsq(X_b, y, rcond=-1)
theta_best_svd
# %% [markdown]
"""
This function computes $\theta = X^+y$, where $X^+$ is the pseudoinverse
of X (specifically the Moore-Penrose inverse). You can use `np.linalg.pinv()`
to compute the pseudoinverse directly:
"""
# %%
ln.pinv(X_b) @ y
# %%
