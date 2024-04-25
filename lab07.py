import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification, load_iris, make_regression, fetch_california_housing
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, KDTree
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

class KNN:
    def __init__(self, n_neighbors=1, use_kd_tree=False):
        self.n_neighbors = n_neighbors
        self.use_kd_tree = use_kd_tree
        self.model = None

    def fit(self, X, y):
        if self.use_kd_tree:
            self.model = KDTree(X)
        else:
            self.model = X, y

    def predict(self, X):
        if self.use_kd_tree:
            _, indices = self.model.query(X, k=self.n_neighbors)
        else:
            X_train, y_train = self.model
            distances = np.sqrt(np.sum((X[:, np.newaxis] - X_train) ** 2, axis=2))
            indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        if len(indices.shape) == 1:
            indices = np.expand_dims(indices, axis=0)

        predictions = np.array([np.argmax(np.bincount(y_train[neighborhood])) for neighborhood in indices])
        return predictions

    def score(self, X, y):
        if isinstance(y[0], (int, np.integer)):
            return accuracy_score(y, self.predict(X))
        else:
            return mean_squared_error(y, self.predict(X))

class KNNClassifier:
    def __init__(self, n_neighbors=1, use_kd_tree=True):
        self.n_neighbors = n_neighbors
        self.use_kd_tree = use_kd_tree
        if self.use_kd_tree:
            self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm='kd_tree')
        else:
            self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_test, y_test):
        return accuracy_score(y_test, self.predict(X_test))

class KNNRegressor:
    def __init__(self, n_neighbors=1, use_kd_tree=True):
        self.n_neighbors = n_neighbors
        self.use_kd_tree = use_kd_tree
        if self.use_kd_tree:
            self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, algorithm='kd_tree')
        else:
            self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_test, y_test):
        return mean_squared_error(y_test, self.predict(X_test))


X, y = make_classification(n_samples=100,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           random_state=3)

# klasyfikacja
knn_classifier = KNNClassifier(n_neighbors=3)
knn_classifier.fit(X, y)
y_pred = knn_classifier.predict(X)
accuracy = knn_classifier.score(X, y)
print("Classification accuracy:", accuracy)
print("____________________________")
# regresja
knn_regressor = KNNRegressor(n_neighbors=3)
knn_regressor.fit(X, y)
y_pred_reg = knn_regressor.predict(X)
reg = knn_regressor.score(X, y)
print("Regression MSE:", reg)


plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.rainbow)
plt.title('Data Visualization')
plt.show()


iris = load_iris()
X_iris, y_iris = iris.data, iris.target

def plot_decision_boundary(model, X, y):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.rainbow)
    plt.show()


model = KNN(n_neighbors=1)
model.fit(X, y)
plot_decision_boundary(model, X, y)


X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=20, random_state=3)
k_values = [1, 3, 5, 7, 9]
results = {'k': [], 'mean_squared_error': []}
# leave one out kroswalidacja
def leave_one_out_cv(X, y, n_neighbors_range):
    loo = LeaveOneOut()
    for n_neighbors in n_neighbors_range:
        accuracy_scores = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = KNN(n_neighbors=n_neighbors)
            model.fit(X_train, y_train)
            accuracy_scores.append(model.score(X_test, y_test))
        print(f"n_neighbors={n_neighbors}: Mean Accuracy = {np.mean(accuracy_scores)}")
print("____________________________")
leave_one_out_cv(X_iris, y_iris, n_neighbors_range=[1, 2, 3, 4, 5])

def compare_execution_time(X, y, n_samples=1000, n_features=10):
    X_comp, y_comp = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)

    def execution_time(model):
        start_time = time.time()
        model.fit(X_comp, y_comp)
        end_time = time.time()
        return end_time - start_time

    basic_time = execution_time(KNN(n_neighbors=3))
    kd_tree_time = execution_time(KNN(n_neighbors=3, use_kd_tree=True))
    print(f"Basic version: {basic_time} seconds")
    print(f"Version using KD-tree: {kd_tree_time} seconds")
print("____________________________")
compare_execution_time(X_iris, y_iris)
print("____________________________")

X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=20, random_state=3)
knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(X_reg, y_reg)
y_pred_reg = knn_regressor.predict(X_reg)
reg1 = mean_squared_error(y_reg, y_pred_reg)
print("Regression:", reg1)
print("____________________________")
plt.scatter(X_reg, y_reg, color='blue', label='Training Data')
plt.scatter(X_reg, y_pred_reg, color='red', label='Model Prediction', marker='x')
plt.title('KNN Regression')
plt.legend(['Training Data', 'Model Prediction'])
plt.show()

# pca
def plot_decision_boundary_pca(model, X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    h = .02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))

    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, alpha=0.8)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=plt.cm.rainbow)
    plt.show()


model = KNN(n_neighbors=1)
model.fit(X, y)
plot_decision_boundary_pca(model, X, y)


X, y = fetch_california_housing(return_X_y=True)
knn_regressor = KNeighborsRegressor()
k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
results = {'k': [], 'mean_squared_error': []}
for k in k_values:
    knn_regressor.n_neighbors = k
    scores = cross_val_score(knn_regressor, X, y, cv=10, scoring='neg_mean_squared_error')
    mean_score = -scores.mean()
    results['k'].append(k)
    results['mean_squared_error'].append(mean_score)

results_df = pd.DataFrame(results)
print(results_df)
