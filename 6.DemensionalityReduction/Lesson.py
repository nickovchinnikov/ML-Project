# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import (
    PCA,
    IncrementalPCA,
    KernelPCA
)
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.datasets import (
    fetch_openml,
    make_swiss_roll
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# %% [markdown]
# ### Let's build a simple 3D dataset
# %%
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
angles

X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

X[:5]
# %% [markdown]
# ### Let's use PCA to reduce the dimensionality of the dataset
# %%
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)

X2D[:5], pca.components_

# %% [markdown]
# ### Variance Ratio
# %%
pca.explained_variance_ratio_

# %%
# Recover the original data
X_new = pca.inverse_transform(X2D)
X_new[:5]
# %% [markdown]
# ### Choose the number of dimensions to keep
# %%
pca = PCA()
pca.fit(X)

cumsum = np.cumsum(pca.explained_variance_ratio_)

d = np.argmax(cumsum >= 0.95) + 1
d

# %% [markdown]
# ### You can choose the ratio of variance to keep
# %%
pca = PCA(n_components=0.95)
pca.fit(X)

len(pca.components_)
# %% [markdown]
# ### Use mnist dataset to find dimensionality reduction
# %%
mnist = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
mnist.target = mnist.target.astype(np.uint8)

# %%
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    mnist.data,
    mnist.target,
    test_size=0.2,
    random_state=42
)

# %%
pca = PCA()
pca.fit(X_train)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

d
# %%
# Plot the cumulative variance ratio
plt.figure(figsize=(10, 5))

plt.plot(cumsum, linewidth=2)

plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

plt.plot(d, 0.95, 'ko')

plt.annotate(
    '95%',
    xy=(d, 0.95),
    xytext=(d + 20, 0.8),
    arrowprops=dict(arrowstyle='->'),
    fontsize=16
)

plt.annotate(
    'Elbow',
    xy=(40, 0.8),
    xytext=(80, 0.7),
    arrowprops=dict(arrowstyle='->'),
    fontsize=16
)

plt.plot([d, d], [0, 0.95], 'k--')
plt.plot([0, d], [0.95, 0.95], 'k--')

plt.axis([0, 400, 0, 1])
plt.grid(True)

plt.show()

# %%
# PCA with the chosen number of components as the ratio of variance to keep
pca = PCA(n_components=0.95)
pca.fit(X_train)

# %%
# The same result as above
pca.n_components_

# %%
np.sum(pca.explained_variance_ratio_)

# %%
# Reduce the dimensionality of the training set
X_reduced = pca.transform(X_train)
X_reduced.shape

# %%
# Restore the original data
X_restored = pca.inverse_transform(X_reduced)
X_restored.shape

# %%
# Random PCA
rnd_pca = PCA(n_components=154, svd_solver="randomized")
rnd_pca.fit(X_train)
X_reduced = rnd_pca.fit_transform(X_train)
X_reduced.shape

# %%
# Incremental PCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)
X_reduced.shape

# %%
# With np.memmap to save the memory. Run in write mode.
filename = "my_mnist.data"
m, n = X_train.shape

X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m, n))
X_mm[:] = X_train

# %%
# Deleting the memmap file triggers the garbage collector to save data to disk
del X_mm

# %%
# Run np.memmap in read mode
X_mm = np.memmap(filename, dtype='float32', mode='readonly', shape=(m, n))

batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)

# %% [markdown]
# ### Kernel PCA
# %%
# Make swiss roll data
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

# %%
rbf_pca = KernelPCA(n_components=154, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

X_reduced[:1]

# %% [markdown]
# ### Swiss roll reduced to 2 dimensions
# Graphical illustration of the data with reduced dimensionality
# Compare different dimensionality reduction methods
# %%
lin_pca = KernelPCA(
    n_components=2, 
    kernel="linear", 
    fit_inverse_transform=True
)
rbf_pca = KernelPCA(
    n_components=2,
    kernel="rbf",
    gamma=0.0433,
    fit_inverse_transform=True
)
sig_pca = KernelPCA(
    n_components=2,
    kernel="sigmoid",
    gamma=0.001,
    coef0=1,
    fit_inverse_transform=True
)

subplots = (
    (131, lin_pca, "Linear kernel"),
    (132, rbf_pca, "RBF kernel, $\gamma=0.04$"),
    (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")
)

plt.figure(figsize=(11, 4))

for subplot, pca, title in subplots:
    X_reduced = pca.fit_transform(X)

    ax = plt.subplot(subplot)
    
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)

    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)

    ax.set_aspect("equal")
    plt.title(title, fontsize=14)
    plt.grid(True)
    plt.tight_layout()

plt.show()

# %% [markdown]
# ### Select a Kernel and Tuning Hyperparameters
# %%
y = t > 6.9

clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression(solver="lbfgs"))
])

param_grid = {
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"]
}

grid_search = GridSearchCV(
    clf,
    param_grid=param_grid,
    cv=3
)

grid_search.fit(X, y)

# %%
# The best hyperparameters
grid_search.best_params_

# %%
# Unsupervised approach
rbf_pca = KernelPCA(
    n_components=2,
    kernel="rbf",
    gamma=0.0433,
    # Very important to use for mapping kernel to the original space 
    fit_inverse_transform=True
)

X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

mean_squared_error(X, X_preimage)

# %% [markdown]
# ### Local Linear Embedding (LLE)
# %%
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

# %%
lle = LocallyLinearEmbedding(
    n_components=2,
    n_neighbors=10,
    random_state=42
)

X_reduced = lle.fit_transform(X)

# %%
# Plot the unrolled swiss roll using the LLE embedding
plt.title("Unrolled swiss roll using LLE", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)

plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)

plt.tight_layout()
plt.grid(True)

plt.show()

# %%
