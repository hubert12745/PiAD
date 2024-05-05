import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def diste(X, C):
    """
    Calculates the Euclidean distance between two sets of points X and C.

    Parameters:
    X : numpy.ndarray
        Data matrix of shape (n_samples, n_features).
    C : numpy.ndarray
        Matrix of centroids of shape (n_centroids, n_features).

    Returns:
    dist : numpy.ndarray
        Euclidean distance matrix of shape (n_samples, n_centroids).
    """
    dist = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
    return dist


def distm(X, C, V):
    """
    Calculates the Mahalanobis distance between two sets of points X and C.

    Parameters:
    X : numpy.ndarray
        Data matrix of shape (n_samples, n_features).
    C : numpy.ndarray
        Matrix of centroids of shape (n_centroids, n_features).
    V : numpy.ndarray
        Covariance matrix of shape (n_features, n_features).

    Returns:
    dist : numpy.ndarray
        Mahalanobis distance matrix of shape (n_samples, n_centroids).
    """
    diff = X[:, np.newaxis, :] - C
    dist = np.sqrt(np.einsum('ijk,kl,ijl->ij', diff, np.linalg.inv(V), diff))
    return dist


def ksrodki(X, k, dist_func=diste, max_iter=100, tol=1e-4):
    """
    K-means clustering algorithm.

    Parameters:
    X : numpy.ndarray
        Data matrix of shape (n_samples, n_features).
    k : int
        Number of clusters.
    dist_func : function, optional
        Function to calculate distance. Default is Euclidean distance.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    tol : float, optional
        Tolerance to declare convergence. Default is 1e-4.

    Returns:
    centroids : numpy.ndarray
        Centroids of shape (k, n_features).
    labels : numpy.ndarray
        Labels of each sample.
    """
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iter):
        dist_matrix = dist_func(X, centroids)
        labels = np.argmin(dist_matrix, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.allclose(centroids, new_centroids, atol=tol):
            break

        centroids = new_centroids

    return centroids, labels


def clustering_quality(X, centroids, labels):
    """
    Computes the clustering quality measure WCSS for K-means clustering.

    Parameters:
    X : numpy.ndarray
        Data matrix of shape (n_samples, n_features).
    centroids : numpy.ndarray
        Centroids of shape (k, n_features).
    labels : numpy.ndarray
        Labels of each sample.

    Returns:
    quality : float
        Clustering quality measure.
    """
    k = centroids.shape[0]
    quality = 0

    for i in range(k):
        dist_within = np.sum(np.linalg.norm(X[labels == i] - centroids[i], axis=1) ** 2)
        quality += dist_within

    quality /= X.shape[0]
    return quality


# Example data (replace with your dataset)
# np.random.seed(42)
# X = np.random.rand(100, 2)
df = pd.read_csv('autos.csv')
df = df[['horsepower', 'price']].dropna()
X = df.to_numpy()
# Example of usage
k = 3
centroids, labels = ksrodki(X, k)
quality = clustering_quality(X, centroids, labels)
print("Clustering quality:", quality)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red', s=200, label='Centroids')
plt.legend()
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
