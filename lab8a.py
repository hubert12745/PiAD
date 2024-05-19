import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def diste(X, C):
    dist = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
    return dist


def distm(X, C, V):
    diff = X[:, np.newaxis, :] - C
    dist = np.sqrt(np.einsum('ijk,kl,ijl->ij', diff, np.linalg.inv(V), diff))
    return dist


def ksrodki(X, k, dist_func=diste, max_iter=100, tol=1e-4):
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
    k = centroids.shape[0]
    quality = 0

    for i in range(k):
        dist_within = np.sum(np.linalg.norm(X[labels == i] - centroids[i], axis=1) ** 2)
        quality += dist_within

    quality /= X.shape[0]
    return quality


def quality(X, clusters, labels):
    # Calculate the numerator
    numerator = np.sum(diste(X, clusters)[np.arange(X.shape[0]), labels])

    # Calculate the denominator
    denominator = np.sum([np.sum(diste(x.reshape(1, -1), clusters[cluster].reshape(1, -1)) ** 2)
                          for x, cluster in zip(X, labels)])

    # Return the ratio, or 0 if the denominator is 0
    return numerator / denominator if denominator != 0 else 0


df = pd.read_csv('autos.csv')
df = df[['horsepower', 'price']].dropna()
X = df.to_numpy()

k = 3
centroids, labels = ksrodki(X, k)
cquality = clustering_quality(X, centroids, labels)
print("Clustering quality:", cquality)
qual = quality(X, centroids, labels)
print("Quality:", qual)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red', s=200, label='Centroids')
plt.legend()
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

data = pd.read_csv('autos.csv')
# data.dropna(inplace=True)
# Check for missing values
# print(data.isnull().sum())

# Handle missing values
encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = encoder.fit_transform(data[column])

# Here we are filling missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Convert non-numeric values to numeric ones
# Here we are using label encoding

print(data.to_string())

k = 3
centroids, labels = ksrodki(data.to_numpy(), k)
cquality = clustering_quality(data.to_numpy(), centroids, labels)
print("Clustering quality:", cquality)
qual = quality(X, centroids, labels)
print("Quality:", qual)

