# %%
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%
# Load the iris dataset
iris = datasets.load_iris()
list(iris.keys())

# %%
# Description of the dataset
print(iris.DESCR)

# %% [markdown]
"""
Loadings the iris dataset into the variables X and y.
"""

# %%
X = iris["data"][:, 2:] # petal length and width
y = iris["target"]

X[:5, :], y[:5]

# %% [markdown]
"""
We need to add bias term for every instance $x_0 = 1$
"""

# %%
X_with_bias = np.c_[np.ones((len(X), 1)), X]

X_with_bias[:5, :]

# %% [markdown]
"""
Set the seed to 2042 to get the same random numbers every time so the output will be the same every time.
"""

# %%
np.random.seed(2042)

# %% [markdown]
"""
Manual implemetation train_test_split
"""
# %%
test_ratio = 0.2
validation_ratio = 0.2

total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size

rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]

X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]

# %%
def to_one_hot(y):
    n_classes = y.max() + 1
    m = len(y)
    Y_one_hot = np.zeros((m, n_classes))
    Y_one_hot[np.arange(m), y] = 1
    return Y_one_hot

# %%
# First 10 instances
y_train[:10]

# %%
# Test to_one_hot to the first 10 instances
to_one_hot(y_train[:10])

# %%
# Let's create the target class probabilities for the test set and the training set
Y_train_one_hot = to_one_hot(y_train)
Y_test_one_hot = to_one_hot(y_test)
Y_valid_one_hot = to_one_hot(y_valid)

# %% [markdown]
"""
Let's implement the softmax regression function by the following equation:

$$s_k(x)=x^T\theta^{(k)}$$

$$\hat{p_k} = \sigma\left(\mathbf{s}(\mathbf{x})\right)_k = \dfrac{\exp\left(s_k(\mathbf{x})\right)}{\sum\limits_{j=1}^{K}{\exp\left(s_j(\mathbf{x})\right)}}$$
"""

# %%
def softmax(x, theta):
    logits = x @ theta
    exps = np.exp(logits)
    exp_sums = np.sum(exps, axis=1, keepdims=True)
    return exps / exp_sums

# %%
n_inputs = X_train.shape[1]
n_outputs = len(np.unique(y_train))

n_inputs, n_outputs

# %% [markdown]
"""
So the equations we will need are the cost (entropy loss) function:

$$J(\mathbf{\Theta}) = - \dfrac{1}{m}\sum\limits_{i=1}^{m}\sum\limits_{k=1}^{K}{y_k^{(i)}\log\left(\hat{p}_k^{(i)}\right)}$$

And the equation for the gradients:

$$\nabla_{\mathbf{\theta}^{(k)}} \, J(\mathbf{\Theta}) = \dfrac{1}{m} \sum\limits_{i=1}^{m}{ \left ( \hat{p}^{(i)}_k - y_k^{(i)} \right ) \mathbf{x}^{(i)}}$$

Note that $\log\left(\hat{p}_k^{(i)}\right)$ may not be computable if $\hat{p}_k^{(i)} = 0$. So we will add a tiny value $\epsilon$ to $\log\left(\hat{p}_k^{(i)}\right)$ to avoid getting `nan` values.
"""

# %%
# Machine eps
eps = np.finfo(np.float32).eps

def training(y_k, x, theta, n_iterations, eta, epsilon = eps):
    m = len(x)
    for iteration in range(n_iterations):
        p_k = softmax(X_train, theta)

        if iteration % 500 == 0:
            entropy_loss = -np.mean(np.sum(y_k * np.log(p_k + epsilon), axis=1))
            print(iteration, entropy_loss)

        grad = (1/m) * x.T @ (p_k - y_k)
        theta = theta - eta * grad
    return theta

# %%
# Assign values to named variables
y_k = Y_train_one_hot
x = X_train
# For the batch gradient it should be the random batch
theta = np.random.randn(n_inputs, n_outputs)
# Additional parameters fot the training
n_iterations = 5001
eta = 0.01

theta = training(y_k, x, theta, n_iterations, eta)
theta

# %% [markdown]
"""
#### Equation 4-21. Softmax Regression classifier prediction

$$
\hat{y} = 
\underset{k}{\operatorname{argmax}} \space \sigma(s(x))_k
$$

Because $\sigma(s(x))_k$ is the ration we can use

$$
\hat{y} =
\underset{k}{\operatorname{argmax}} \space s_k(x)
$$

Replace $s_k(x)$

$$
\hat{y} =
\underset{k}{\operatorname{argmax}} \space ((\theta^{(k)})^T x)
$$
"""

# %%
# Check on the validation set
Y_proba = softmax(X_valid, theta)
y_predict = np.argmax(Y_proba, axis=1)

accuracy = (y_valid == y_predict).astype(np.int32)

accuracy_score = np.mean(accuracy)
accuracy_score

# %% [markdown]
"""
Create traning function with $l_2$ regularization

#### Equation 4-8. Ridge Regression cost function with $l_2$ regularization

$$
J(\theta)=MSE(\theta) + \alpha \frac{1}{2} \sum_{i=1}^{n} {\theta_i^2}
$$
"""

# %%
# $l_2$ loss example
l2_loss = 1/2 * np.sum(np.square(theta[1:]))
l2_loss

# %%
def training_reg_l2(y_k, x, theta, n_iterations, eta, alpha, epsilon = eps):
    m = len(x)
    for iteration in range(n_iterations):
        p_k = softmax(X_train, theta)

        if iteration % 500 == 0:
            xentropy_loss = -np.mean(np.sum(y_k * np.log(p_k + epsilon), axis=1))
            l2_loss = alpha * 1/2 * np.sum(np.square(theta[1:]))
            entropy_loss = xentropy_loss + l2_loss
            print(iteration, entropy_loss)

        # additional l_2 parameter
        reg = np.r_[np.zeros([1, n_outputs]), 0.1 * theta[1:]]
        grad = (1/m) * x.T @ (p_k - y_k) + reg
        theta = theta - eta * grad
    return theta

# %%
# Assign values to named variables
y_k = Y_train_one_hot
x = X_train
# For the batch gradient it should be the random batch
theta = np.random.randn(n_inputs, n_outputs)
# Additional parameters fot the training
n_iterations = 5001
eta = 0.1
alpha = 0.1

theta = training_reg_l2(y_k, x, theta, n_iterations, eta, alpha)
theta

# %%
# Check performance of the model
Y_proba = softmax(X_valid, theta)
y_predict = np.argmax(Y_proba, axis=1)

accuracy = (y_valid == y_predict).astype(np.int32)

accuracy_score = np.mean(accuracy)
accuracy_score

# %%
# Let's add early stopping
def training_l2_early_stop(y_k, x, theta, n_iterations, eta, alpha, r_num=6, epsilon = eps):
    best_loss = np.infty
    m = len(x)
    for iteration in range(n_iterations):
        p_k = softmax(X_train, theta)

        # additional l_2 parameter
        reg = np.r_[np.zeros([1, n_outputs]), 0.1 * theta[1:]]
        grad = (1/m) * x.T @ (p_k - y_k) + reg
        theta = theta - eta * grad

        xentropy_loss = -np.mean(np.sum(y_k * np.log(p_k + epsilon), axis=1))
        l2_loss = alpha * 1/2 * np.sum(np.square(theta[1:]))
        entropy_loss = xentropy_loss + l2_loss

        if iteration % 500 == 0:
            print(iteration, entropy_loss)
        
        if round(entropy_loss, r_num) < round(best_loss, r_num):
            best_loss = entropy_loss
        else:
            print(iteration - 1, best_loss)
            print(iteration, entropy_loss, "early stopping!")
            return theta

    return theta

# %%
# Assign values to named variables
y_k = Y_train_one_hot
x = X_train
# For the batch gradient it should be the random batch
theta = np.random.randn(n_inputs, n_outputs)
# Additional parameters fot the training
n_iterations = 5001
eta = 0.1
alpha = 0.1

theta = training_l2_early_stop(y_k, x, theta, n_iterations, eta, alpha)
theta

# %%
# Check performance of the model
Y_proba = softmax(X_valid, theta)
y_predict = np.argmax(Y_proba, axis=1)

accuracy = (y_valid == y_predict).astype(np.int32)

accuracy_score = np.mean(accuracy)
accuracy_score

# %%
# Plot the model of the whole prediction
x0, x1 = np.meshgrid(
    np.linspace(0, 8, 500).reshape(-1, 1),
    np.linspace(0, 3.5, 200).reshape(-1, 1),
)
X_new = np.c_[x0.ravel(), x1.ravel()]
X_new_with_bias = np.c_[np.ones([len(X_new), 1]), X_new]

Y_proba = softmax(X_new_with_bias, theta)
y_predict = np.argmax(Y_proba, axis=1)

zz1 = Y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()
# %%
