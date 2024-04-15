import numpy as np
from sklearn.neighbors import KDTree
from sklearn.datasets import make_classification


def euclidean_distance(point, data):
    return np.sqrt(np.sum(point - data) ** 2)


class knnAlgorithm:
    def __init__(self, n_neighbors=1, use_KDTree=False):
        self.n_neighbors = n_neighbors
        self.use_KDTree = use_KDTree
        self.x_train = None
        self.y_train = None
        if self.use_KDTree:
            self.kdTree = KDTree(X)

    def fit(self, X, y):
        self.x_train = X
        self.y_train = y

    def predict(self, X):
        neighbors = []
        for x in X:
            distances = euclidean_distance(x, self.x_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.n_neighbors])
        return neighbors

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = sum(y == y_pred) / len(y)
        return accuracy


X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=3
)

knn = knnAlgorithm(n_neighbors=3,use_KDTree=True)
knn.fit(X, y)
print("KNN Accuracy: ", knn.score(X, y))
