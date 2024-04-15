import numpy as np
from sklearn.neighbors import KDTree
from sklearn.datasets import make_classification


class knnAlgorithm:
    def __init__(self, n_neighbors=1, use_KDTree=False):
        self.n_neighbors = n_neighbors
        self.use_KDTree = use_KDTree
        self.X_Train = None
        self.Y_Train = None
        if self.use_KDTree:
            self.kdTree = KDTree(X)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        if self.use_KDTree:
            _, indices = self.kdTree.query(X, k=self.n_neighbors)
        else:
            distances = np.sqrt(np.sum((X[:, np.newaxis] - self.X_train) ** 2, axis=2))
            indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        if len(self.y_train.shape) == 1:
            predictions = np.array([np.argmax(np.bincount(self.y_train[neighbors])) for neighbors in indices])
        else:
            predictions = np.array([np.mean(self.y_train[neighbors], axis=0) for neighbors in indices])
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        if len(y.shape) == 1:
            return np.mean((predictions - y) ** 2)
        else:
            return np.mean(predictions == y)


X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=3
)
