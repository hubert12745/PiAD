import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
import time

class KNNAlgorithm:
    def __init__(self, n_neighbors=1, use_kd_tree=False):
        self.n_neighbors = n_neighbors
        self.use_kd_tree = use_kd_tree
        self.x_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X, y):
        self.x_train = X
        self.y_train = y
        if self.use_kd_tree:
            self.tree = KDTree(X)

    def predict(self, X):
        if self.use_kd_tree:
            _, indices = self.tree.query(X, k=self.n_neighbors)
            knn_indices = indices
        else:
            knn_indices = self.find_neighbors(X)

            return self.classify(knn_indices)


    def find_neighbors(self, X):
        distances = np.sqrt(np.sum((X[:, np.newaxis] - self.x_train) ** 2, axis=2))
        return np.argsort(distances, axis=1)[:, :self.n_neighbors]

    def classify(self, knn_indices):
        y_train_int = self.y_train.astype(int)
        knn_indices_int = knn_indices.astype(int)
        modes = []
        for neighbors in y_train_int[knn_indices_int]:
            neighbors = neighbors[neighbors >= 0]
            if len(neighbors) > 0:
                modes.append(np.argmax(np.bincount(neighbors)))
            else:
                modes.append(0)
        return np.array(modes)

    def regress(self, knn_indices):
        return np.mean(self.y_train[knn_indices], axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        if len(y.shape) == 1:  # Regresja
            return 1 - np.mean((y_pred - y) ** 2)
        else:  # Klasyfikacja
            return np.mean(y_pred == y)

def plot_decision_boundary(knn_model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, colors='blue', linestyles='-')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k')
    plt.title(title)
    plt.show()


X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=3
)


knn = KNNAlgorithm(n_neighbors=3, use_kd_tree=False)
knn.fit(X, y)
plot_decision_boundary(knn, X, y, 'Klasyfikacja k-NN')

X_iris, y_iris = load_iris(return_X_y=True)


pca = PCA(n_components=2)
X_iris_pca = pca.fit_transform(X_iris)

knn_iris = KNNAlgorithm(n_neighbors=3, use_kd_tree=False)
knn_iris.fit(X_iris_pca, y_iris)
plot_decision_boundary(knn_iris, X_iris_pca, y_iris, 'Klasyfikacja k-NN z PCA')

# Krzyżowa walidacja Leave-One-Out
loo = LeaveOneOut()
results = []

for k in range(1, 11):
    knn_cv = KNNAlgorithm(n_neighbors=k, use_kd_tree=False)
    scores = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn_cv.fit(X_train, y_train)
        score = knn_cv.score(X_test, y_test)
        scores.append(score)

    mean_score = np.mean(scores)
    results.append((k, mean_score))

print("Wyniki:")
print("k\tŚrednia ocena")
for k, mean_score in results:
    print(f"{k}\t{mean_score}")

# Porównanie wydajności
X_example, y_example = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_repeated=0,
    random_state=42
)

start_basic = time.time()
knn_basic = KNNAlgorithm(n_neighbors=3, use_kd_tree=False)
knn_basic.fit(X_example, y_example)
end_basic = time.time()
time_basic = end_basic - start_basic

start_kdtree = time.time()
knn_kdtree = KNNAlgorithm(n_neighbors=3, use_kd_tree=True)
knn_kdtree.fit(X_example, y_example)
end_kdtree = time.time()
time_kdtree = end_kdtree - start_kdtree

print(f"Czas wersji podstawowej: {time_basic} sekund")
print(f"Czas wersji z KD-Drzewami: {time_kdtree} sekund")
