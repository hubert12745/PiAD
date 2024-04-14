from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
def wiPCA(data, n_components):
    mean = np.mean(data, axis=0)
    data -= mean
    # covMatrix = np.dot(data.T, data) / n_components
    covMatrix = np.cov(data, rowvar=False)
    eigenVals, eigenVect = np.linalg.eig(covMatrix)
    principalComponents = eigenVect[:, 0]
    resultData = np.dot(data, principalComponents)
    return resultData, principalComponents

data = np.random.randn(200, 2)
N = len(data)
result, principalComponent = wiPCA(data, N)

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.scatter(result, result, alpha=0.5)
plt.quiver(0, 0, principalComponent[0], principalComponent[1], scale=3, color='r')
plt.tight_layout()
plt.show()

def wiPCA_Iris(data, n_components):
    mean = np.mean(data, axis=0)
    data -= mean
    covMatrix = np.cov(data, rowvar=False)
    eigenVals, eigenVect = np.linalg.eig(covMatrix)
    idx = np.argsort(eigenVals)[::-1]
    principalComponents = eigenVect[:, idx[:2]]
    resultData = np.dot(data, principalComponents)
    return resultData, principalComponents


data = load_iris()
X = data.data
y = data.target
N = len(X)

result, principalComponents = wiPCA_Iris(X, N)

plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(y)):
    plt.scatter(result[y == label, 0], result[y == label, 1], label=f'Klasa {label}', alpha=0.5)


plt.title('PCA dla zbioru danych Iris')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


digits = load_digits()
X = digits.data
y = digits.target

N = X.shape[0]
D = X.shape[1]

result, principalComponents = wiPCA_Iris(X, N)

pca = PCA()
pca.fit(X)
varianceRatio = pca.explained_variance_ratio_
totalVarianceRatio = np.cumsum(varianceRatio)

plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, D + 1), totalVarianceRatio, marker='o')
plt.xlabel('numer składowej')
plt.ylabel('skumulowana wariancja')
plt.title('Krzywa wariancji')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
for i in range(10):
    plt.scatter(result[y == i, 0], result[y == i, 1], label=str(i), alpha=0.5)
plt.xlabel('składowa 1')
plt.ylabel('skladowa 2')
plt.title('PCA dla zbioru danych digits')
plt.colorbar()
plt.show()

reconstructedX = np.dot(result, principalComponents.T) + np.mean(X, axis=0)

mse = mean_squared_error(X, reconstructedX)
print("Średni błąd rekonstrukcji:", mse)