import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def diste(X, C):
    if X.ndim == 1:
        X = X.reshape(1, -1) # Upewnia się, że X jest tablicą 2D
    if C.ndim == 1:
        C = C.reshape(1, -1) # Upewnia się, że C jest tablicą 2D

    # Oblicza odległość euklidesową między każdym punktem w X a każdym centroidem w C
    dist = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
    return dist


def distm(X, C, V):
    diff = X[:, np.newaxis, :] - C # Oblicza różnicę między punktami a centroidami

    dist = np.sqrt(np.einsum('ijk,kl,ijl->ij', diff, np.linalg.inv(V), diff))
    return dist


def ksrodki(X, k, dist_func=diste, max_iter=100, tol=1e-4):
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iter):
        dist_matrix = dist_func(X, centroids) # Oblicza macierz odległości
        labels = np.argmin(dist_matrix, axis=1) # Przypisuje etykiety najbliższego centroidu
        # Oblicza nowe centroidy jako średnie punktów przypisanych do każdego centroidu
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.allclose(centroids, new_centroids, atol=tol): # Sprawdza konwergencję
            break

        centroids = new_centroids # Aktualizuje centroidy

    return centroids, labels # Zwraca ostateczne centroidy i etykiety

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    categorical_features = data.select_dtypes(include=['object']).columns
    numerical_features = data.select_dtypes(exclude=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Uzupełnia brakujące wartości średnią
        ('scaler', StandardScaler())]) # Standaryzuje dane

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # Uzupełnianie brakujących wartości
        ('onehot', OneHotEncoder(handle_unknown='ignore'))]) # Kodowanie one-hot

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),  # Przetwarzanie cech numerycznych
            ('cat', categorical_transformer, categorical_features)]) # Przetwarzanie cech kategorycznych

    X = preprocessor.fit_transform(data)
    return X

def visualize_with_pca(X, labels, centroids):
    pca = PCA(n_components=2) # Redukcja wymiarów do 2D
    principal_components = pca.fit_transform(X) # Transformuje dane
    centroid_pca = pca.transform(centroids) # Transformuje centroidy

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, alpha=0.6) # Rysuje punkty danych z kolorami odpowiadającymi etykietom klastrów
    plt.scatter(centroid_pca[:, 0], centroid_pca[:, 1], marker='X', color='red', s=200, label='Centroids') # Rysuje centroidy z wyróżniającym się markerem
    plt.xlabel(f'Principal Component 1')
    plt.ylabel(f'Principal Component 2')
    plt.colorbar(scatter)
    plt.legend()
    plt.title('PCA - Cluster Visualization with Centroids')
    plt.show()

def clustering_quality(X, labels, centroids):
    K = centroids.shape[0]
    inter_cluster_distances = np.sum([np.linalg.norm(centroids[k] - centroids[l]) for k in range(K) for l in range(k + 1, K)]) # Oblicza odległości między centroidami
    intra_cluster_distances = np.sum([np.linalg.norm(point - centroids[labels[i]])**2 for i, point in enumerate(X)])# Obliczanie odległości wewnątrz klastrów
    return inter_cluster_distances / intra_cluster_distances if intra_cluster_distances != 0 else 0 # Zwracanie stosunku odległości międzyklastrowych do odległości wewnątrzklastrowych

data_path = 'autos.csv'
X = load_and_preprocess_data(data_path)
k = 4
centroids, labels = ksrodki(X, k)
print("Clusters formed.")
visualize_with_pca(X, labels, centroids)
quality = clustering_quality(X, labels, centroids)
print("Clustering Quality:", quality)