# %%
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.colors import LogNorm

from sklearn.datasets import (
    make_blobs,
    load_digits,
    make_moons
)
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    DBSCAN
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.mixture import (
    GaussianMixture,
    BayesianGaussianMixture
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.pipeline import Pipeline

# %%
# Setup center and standard deviation
blob_centers = [
    [0.2, 2.3],
    [-1.5, 2.3],
    [-2.8,  1.8],
    [-2.8,  2.8],
    [-2.8,  1.3]
]
blob_std = [0.4, 0.3, 0.1, 0.1, 0.2]

# %%
# Generate blobs data
X, y = make_blobs(
    n_samples=2000,
    centers=blob_centers,
    cluster_std=blob_std,
    random_state=42
)

# %%
# Plot the data
def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=2)
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.grid(True)

# %%
plt.figure(figsize=(8, 6))
plot_clusters(X)
plt.show()

# %%
# Let's train KMeans with 5 clusters on the data
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

# %%
y_pred

# %%
# The same value
y_pred is kmeans.labels_

# %%
kmeans.cluster_centers_

# %%
kmeans.labels_

# %%
# We can predict the labels for the new data
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)

# %%
def plot_decision_boundary(
    clusterer,
    X,
    resolution=1000,
    show_centroids=True,
    show_xlabel=True,
    show_ylabel=True,
    circle_color='w',
    cross_color='k'
):
    centroids = kmeans.cluster_centers_

    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1

    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                np.linspace(mins[1], maxs[1], resolution))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 4))

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

    if show_centroids:
        plt.scatter(centroids[:, 0], centroids[:, 1],
                        marker='o', s=35, linewidths=8,
                        color=circle_color, zorder=10, alpha=0.9)
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=2, linewidths=12,
                    color=cross_color, zorder=11, alpha=1)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
            cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
            linewidths=1, colors='k')

    if show_xlabel:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabel:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

# %%
plot_decision_boundary(kmeans, X, show_centroids=True)
plt.show()

# %%
# Let's train MiniBatchKMeans with 5 clusters on the data
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, batch_size=3072)
minibatch_kmeans.fit(X)

# %% [markdown]
# ### Finding the Optimal Number of Clusters
# %%
# Inertia decreasing rate visualization
kmeans_per_k = [
    KMeans(n_clusters=k, random_state=42).fit(X)
    for k in range(1, 11)
]

inertias = [
    kmeans.inertia_ for kmeans in kmeans_per_k
]

plt.figure(figsize=(8, 4))

plt.plot(range(1, 11), inertias, marker='o')

plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)

plt.annotate('Elbow',
    xy=(4, inertias[3]),
    xytext=(0.55, 0.55),
    textcoords='figure fraction',
    fontsize=16,
    arrowprops=dict(facecolor='black', shrink=0.1)
)

plt.axis([1, 8.5, 0, 1300])

# %% [markdown]
# ### Let's plot the silhouette score as a function of $k$:
# %%
silhouette_scores = [
    silhouette_score(X, model.labels_)
    for model in kmeans_per_k[1:]
]

# %%
plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), silhouette_scores, 'bo-')
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette Score", fontsize=14)
plt.axis([1.8, 8.5, 0.5, 0.7])
plt.show()

# %% [markdown]
# ### Clustering fot image segmentation
# %%
image_path = os.path.join('.', 'ladybug.png')
image = imread(image_path)

image.shape

# %%
X = image.reshape(-1, 3)
X.shape

kmeans = KMeans(n_clusters=8, random_state=42).fit(X)

# %%
kmeans.cluster_centers_, kmeans.labels_

# %%
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img.shape

# %%
segmented_img = segmented_img.reshape(image.shape)
segmented_img.shape

# %%
segmented_imgs = []
n_colors = tuple(
    reversed(range(2, 11, 2))
)

for n_cluster in n_colors:
    kmeans = KMeans(n_clusters=n_cluster, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[
        kmeans.labels_
    ].reshape(image.shape)
    segmented_imgs.append(segmented_img)

# %%
# Plot the original image and the segmented images
plt.figure(figsize=(10, 5))

plt.subplot(231)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

for idx, n_cluster in enumerate(n_colors):
    segmented_img = segmented_imgs[idx]
    plt.subplot(232 + idx)
    plt.imshow(segmented_img)
    plt.title("k = %d" % n_cluster)
    plt.axis('off')

plt.show()

# %% [markdown]
# ### Using Clustering for Preprocessing
# %%
X_digits, y_digits = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_digits,
    y_digits,
    random_state=42
)

# %%
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# %%
log_reg.score(X_test, y_test)

# %% [markdown]
# The Logistic Regression can't converge on the test set
# so we'll use the PCA to reduce the dimensionality of the data.
# %%
pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# Increase the max_iter to avoid the convergence problems
log_reg2 = LogisticRegression(max_iter=200, random_state=42)
log_reg2.fit(X_train_reduced, y_train)

# Much better!
log_reg2.score(X_test_reduced, y_test)

# %% [markdown]
# Try TSNE to reduce the dimensionality of the data and check the score
# %%
tsne = TSNE(n_components=2, random_state=42)
X_train_reduced2 = tsne.fit_transform(X_train)
X_test_reduced2 = tsne.fit_transform(X_test)

log_reg3 = LogisticRegression(max_iter=200, random_state=42)
log_reg3.fit(X_train_reduced2, y_train)

# Too bad!
log_reg3.score(X_test_reduced2, y_test)

# %%
# Let's use KMeans in a pipeline to cluster the digits
pipeline = Pipeline([
    ('kmeans', KMeans(n_clusters=50)),
    ('log_reg', LogisticRegression(max_iter=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# %%
pipeline.score(X_test, y_test)

# %% [markdown]
# ### Let's tune the params of Logistic Regression
# %%
log_reg = LogisticRegression(
    multi_class="ovr",
    solver="lbfgs",
    max_iter=5000,
    random_state=42
)

log_reg.fit(
    X_train,
    y_train
)

# %%
# With 5000 iterations we get 0.9688
log_reg_score = log_reg.score(X_test, y_test)
log_reg_score

# %%
# Let's create a pipeline with new parameters
pipeline = Pipeline([
    ('kmeans', KMeans(n_clusters=50)),
    ('log_reg', LogisticRegression(
        multi_class="ovr",
        solver="lbfgs",
        max_iter=5000,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# %%
# With 5000 iterations we get 0.9733
pipeline_score = pipeline.score(X_test, y_test)
pipeline_score

# %%
# Drop of the error
1 - (1 - pipeline_score) / (1 - log_reg_score)

# %%
param_grid = dict(
    kmeans__n_clusters=range(2, 100),
)
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)

# %%
grid_clf.best_params_

# %%
grid_clf_score = grid_clf.score(X_test, y_test)
grid_clf_score

# %%
# The final drop of the error
1 - (1 - grid_clf_score) / (1 - pipeline_score)

# %% [markdown]
# ### Clusters for semi-supervised learning
# %%
n_labeled = 50

log_reg = LogisticRegression(max_iter=5000)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])

# %%
# For 50 labeled samples we get 0.8222
log_reg.score(X_test, y_test)

# %%
k = 50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)

representative_digits_idx = np.argmin(X_digits_dist, axis=0)

X_representative_digits = X_train[representative_digits_idx]
y_representative_digits = y_train[representative_digits_idx]

# %%
# Plot the representative digits
plt.figure(figsize=(8, 2))

for idx, digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, idx + 1)
    plt.imshow(digit.reshape(8, 8), cmap='binary', interpolation='bilinear')
    plt.axis('off')

plt.show()

y_representative_digits

# %%
log_reg = LogisticRegression(max_iter=5000)
log_reg.fit(X_representative_digits, y_representative_digits)

log_reg.score(X_test, y_test)

# %%
y_train_propagated = np.empty(len(X_train), dtype=np.int32)

for i in range(k):
    y_train_propagated[
        kmeans.labels_ == i
    ] = y_representative_digits[i]

# %%
log_reg = LogisticRegression(max_iter=5000)
log_reg.fit(X_train, y_train_propagated)

log_reg.score(X_test, y_test)

# %%
# Let's try to use 20% closest digits to the centr of the cluster
percentile_closest = 20

X_digits_dist[0]

X_cluster_dist = X_digits_dist[
    np.arange(len(X_train)),
    kmeans.labels_
]

for i in range(k):
    in_cluster = kmeans.labels_ == i
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_dist = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = X_cluster_dist > cutoff_dist
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)

# Prepare the data for Logistic Regression
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

# %%
log_reg = LogisticRegression(max_iter=5000)
log_reg.fit(
    X_train_partially_propagated,
    y_train_partially_propagated
)

# %%
log_reg.score(X_test, y_test)

# %%
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)

# %%
dbscan.labels_

# %%
len(dbscan.core_sample_indices_)

# %%
dbscan.components_

# %%
dbscan2 = DBSCAN(eps=0.2)
dbscan2.fit(X)

# %% [markdown]
# ### Gaussian mixtures
# %%
X1, y1 = make_blobs(
    n_samples=1000,
    centers=((4, -4), (0, 0)),
    random_state=42
)
X1 = X1.dot(np.array([
    [0.374, 0.95],
    [0.732, 0.598]
]))

X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]

X = np.r_[X1, X2]
y = np.r_[y1, y2]
# %%
X, y
# %%
gm = GaussianMixture(
    n_components=3,
    n_init=10,
    random_state=42
)
gm.fit(X)

# %%
gm.weights_

# %%
gm.means_

# %%
gm.covariances_

# %%
gm.converged_

# %%
gm.n_iter_

# %%
gm.predict(X)

# %%
gm.predict_proba(X)

# %% [markdown]
# #### Gaussian mixtures is generative model
# %%
X_new, y_new = gm.sample(6)

X_new, y_new

# %%
gm.score_samples(X)

# %%
# Let's check that PDF integrates to 1
resolution = 100
x_grid = np.arange(-10, 10, 1 / resolution)

xx, yy = np.meshgrid(x_grid, x_grid)

X_full_grid = np.vstack([xx.ravel(), yy.ravel()]).T

pdf = np.exp(gm.score_samples(X_full_grid))

pdf_probas = pdf * (1 / resolution) ** 2
pdf_probas.sum()

# %%
def plot_gaussian_mixtures(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1

    xx, yy = np.meshgrid(
        np.linspace(mins[0], maxs[0], resolution),
        np.linspace(mins[1], maxs[1], resolution)
    )
    xx_yy = np.c_[xx.ravel(), yy.ravel()]

    Z = -clusterer.score_samples(xx_yy)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, norm=LogNorm(),
        levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z, norm=LogNorm(),
        levels=np.logspace(0, 2, 12), linewidths=1, colors='k')

    Z = clusterer.predict(xx_yy)
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, 
        colors='r', linewidths=2, linestyles='dashed')

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

# %%
plt.figure(figsize=(8, 4))
plot_gaussian_mixtures(gm, X)
plt.show()

# %% [markdown]
# ### Anomaly detection using Gaussian Mixtures
# %%
dencities = gm.score_samples(X)
dencity_treshold = np.percentile(dencities, 4)

anomalies = X[dencities < dencity_treshold]

# %%
# Let's plot the anomalies
plt.figure(figsize=(8, 4))
plot_gaussian_mixtures(gm, X, show_ylabels=False)
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='r', marker='*')
plt.show()

# %%
gm.bic(X), gm.aic(X)

# %%
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)
np.round(bgm.weights_, 2)

# %%
