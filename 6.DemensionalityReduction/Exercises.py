# %%
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# %% [markdown]
# ### Exercise 9
# %%
# Fetch mnist dataset
mnist = fetch_openml(
    'mnist_784',
    version=1,
    as_frame=False,
    cache=True
)
mnist.target = mnist.target.astype(np.uint8)

data, target = mnist.data, mnist.target

# %%
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data,
    target,
    test_size=10000,
    random_state=42
)
# %%
# Train a random forest classifier and measure the time it takes
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

t0 = time.time()
rfc.fit(X_train, y_train)
t1 = time.time()

print('Training time: %.3f sec.' % (t1 - t0))
# %%
# Measure the accuracy on the test set
y_pred = rfc.predict(X_test)
accuracy_score(y_test, y_pred)
# %%
# Use PCA to reduce the dimensionality of the dataset
pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# %%
rfc_pca = RandomForestClassifier(n_estimators=100, random_state=42)

t0 = time.time()
rfc_pca.fit(X_train_reduced, y_train)
t1 = time.time()

print('Training time for pca dataset: %.3f sec.' % (t1 - t0))
# %%
y_pred = rfc_pca.predict(X_test_reduced)
accuracy_score(y_test, y_pred)

# %% [markdown]
# ### Exercise 10
# %%
np.random.seed(42)

m = 10000
idx = np.random.permutation(60000)[:m]

X = mnist['data'][idx]
y = mnist['target'][idx]

# %%
# Let's use t-SNE to reduce the dimensionality of the dataset
tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)

# %%
# Let's use matplotlib to plot a scatter plot of the data using a different color for each digit
plt.figure(figsize=(13, 10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()

# %%
# Let's focus on 2, 3 and 5 which overlap most
overlap_digits = [2, 3, 5]

indexes = np.isin(y, overlap_digits)
c = y[indexes]

plt.figure(figsize=(9, 9))

plt.scatter(
    X_reduced[indexes, 0],
    X_reduced[indexes, 1],
    c=c
)

plt.axis('off')
plt.show()

# %%
# Let's run t-SNE on these digits
idx = (y == 2) | (y == 3) | (y == 5)
X_subset = X[idx]
y_subset = y[idx]

tsne_subset = TSNE(n_components=2, random_state=42)
X_subset_reduced = tsne_subset.fit_transform(X_subset)

# %%
# Let's plot the subset of the data
plt.figure(figsize=(9, 9))

plt.scatter(
    X_subset_reduced[:, 0],
    X_subset_reduced[:, 1],
    c=c
)

plt.axis('off')
plt.show()

# %%
# Let's make a pipeline PCA and t-SNE to speed up the computation
pca_tsne = Pipeline([
    ('pca', PCA(n_components=0.95)),
    ('tsne', TSNE(n_components=2, random_state=42))
])

t0 = time.time()
X_pca_tsne_reduced = pca_tsne.fit_transform(X)
t1 = time.time()
print("PCA + t-SNE took {:.1f}s.".format(t1 - t0))

plt.figure(figsize=(13, 10))
plt.scatter(
    X_pca_tsne_reduced[:, 0],
    X_pca_tsne_reduced[:, 1],
    c=y,
    cmap="jet"
)
plt.axis('off')
plt.colorbar()
plt.show()

plt.show()

# %%
