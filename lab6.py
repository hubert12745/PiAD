import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA


def wiPCA(data, nComponents):
    mean = np.mean(data, axis=0)
    dataCentered = data - mean
    covMatrix = np.cov(dataCentered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covMatrix)
    idx = np.argsort(eigenvalues)[::-1]
    principalComponents = eigenvectors[:, idx[:nComponents]]
    transformedData = np.dot(dataCentered, principalComponents)

    return transformedData, principalComponents


dataRandom = np.random.randn(200, 2)

resultRandom, principalComponentsRandom = wiPCA(dataRandom, nComponents=1)

plt.figure(figsize=(8, 6))
plt.scatter(dataRandom[:, 0], dataRandom[:, 1], alpha=0.5)
plt.scatter(resultRandom, resultRandom, alpha=0.5)
plt.quiver(0, 0, principalComponentsRandom[0, 0], principalComponentsRandom[1, 0], scale=3, color='r')
plt.tight_layout()
plt.show()

dataIris = load_iris()
XIris = dataIris.data
yIris = dataIris.target
resultIris, principalComponentsIris = wiPCA(XIris, nComponents=2)

plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(yIris)):
    plt.scatter(resultIris[yIris == label, 0], resultIris[yIris == label, 1], label=f'Class {label}', alpha=0.5)
plt.title('PCA dla obiektów z bazy iris')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


dataDigits = load_digits()
XDigits = dataDigits.data
yDigits = dataDigits.target
resultDigits, principalComponentsDigits = wiPCA(XDigits, nComponents=2)

pca = PCA()
pca.fit(XDigits)
varianceRatio = pca.explained_variance_ratio_
totalVarianceRatio = np.cumsum(varianceRatio)

plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, XDigits.shape[1] + 1), totalVarianceRatio)
plt.xlabel('numer składowej')
plt.ylabel('skumulowana wariancja')
plt.title('Wizualizacja wariancji składowych głównych')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
for i in range(10):
    plt.scatter(resultDigits[yDigits == i, 0], resultDigits[yDigits == i, 1], label=str(i), alpha=0.5)
plt.xlabel('składowa 1')
plt.ylabel('składowa 2')
plt.title('PCA dla obiektów z bazy digits')
plt.colorbar()
plt.legend()
plt.tight_layout()
plt.show()

reconstructedX = np.dot(resultDigits, principalComponentsDigits.T) + np.mean(XDigits, axis=0)
mse = mean_squared_error(XDigits, reconstructedX)
print("Mean Squared Error:", mse)
