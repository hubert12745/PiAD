import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
import pandas as pd

# Generowanie przykładowych danych
X, y = make_classification(n_samples=1600, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                           n_classes=4)

# Podział danych na zestawy uczące i testowe w proporcji 50/50
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Lista klasyfikatorów bazowych
classifiers = [
    svm.SVC(kernel='linear', probability=True),
    svm.SVC(kernel='rbf', probability=True),
    LogisticRegression(),
    Perceptron()
]

# Tworzenie listy par klasyfikatorów opakowanych w OneVsOneClassifier i OneVsRestClassifier
classifier_pairs = [
    (OneVsOneClassifier(classifier), OneVsRestClassifier(classifier))
    for classifier in classifiers
]


# Funkcja do rysowania wyników klasyfikacji
def plot_results(X_test, y_test, y_pred, title):
    plt.figure(figsize=(15, 5))
    plt.suptitle(title)

    # Oczekiwane
    plt.subplot(1, 3, 1)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', alpha=0.6, edgecolor='k')
    plt.title('oczekiwane')

    # Obliczone
    plt.subplot(1, 3, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', alpha=0.6, edgecolor='k')
    plt.title('obliczone')

    # Różnice
    plt.subplot(1, 3, 3)
    correct = (y_test == y_pred)
    incorrect = ~correct
    plt.scatter(X_test[correct, 0], X_test[correct, 1], c='green', alpha=0.6, edgecolor='k', label='correct')
    plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1], c='red', alpha=0.6, edgecolor='k', label='incorrect')
    plt.legend()
    plt.title('różnice')

    plt.show()


# Funkcja do rysowania krzywych ROC
def plot_roc_curves(y_test, y_score, classifier_name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %d' % (roc_auc[i], i))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for ' + classifier_name)
    plt.legend(loc="lower right")
    plt.show()


# Funkcja do rysowania powierzchni dyskryminacyjnej
def plot_decision_surface(X, y, classifier, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='viridis', edgecolor='k')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def plot_classification_quality(metrics_df):
    metrics_df.T.plot(kind='bar', figsize=(12, 8))
    plt.title('Porównanie jakości klasyfikacji')
    plt.xlabel('Miary jakości')
    plt.ylabel('Wartości miar')
    plt.xticks(rotation=0)
    plt.legend(loc='best')
    plt.show()
# Binarize the output
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]
metrics_list = []
# Przetwarzanie każdego klasyfikatora
for i, (ovo, ovr) in enumerate(classifier_pairs):
    for clf, name in zip([ovo, ovr], ["OneVsOne", "OneVsRest"]):
        # Uczenie klasyfikatora
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_score = clf.decision_function(X_test)

        # Obliczanie miar jakości
        accuracy = metrics.accuracy_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        roc_auc = metrics.roc_auc_score(y_test_bin, y_score, average='macro')

        metrics_list.append([f"{name}={clf.estimator}", accuracy, recall, precision, f1, roc_auc])

        # Wyświetlanie miar jakości
        print(f"{name} Classifier - Pair {i + 1}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"ROC AUC: {roc_auc:.2f}")
        print()

        # Rysowanie wyników klasyfikacji
        plot_results(X_test, y_test, y_pred, f'{name} Classifier - Pair {i + 1}')
        time.sleep(0.5)
        # Rysowanie krzywych ROC
        plot_roc_curves(y_test, y_score, f'{name} Classifier - Pair {i + 1}')
        time.sleep(0.5)
        # Rysowanie powierzchni dyskryminacyjnej
        plot_decision_surface(X_train, y_train, clf, f'{name} Classifier - Pair {i + 1}')

        # Dodanie pauzy między generowaniem wykresów
        time.sleep(0.5)
metrics_df = pd.DataFrame(metrics_list, columns=['Classifier', 'accuracy_score', 'recall_score', 'precision_score', 'f1_score', 'roc_auc'])
metrics_df.set_index('Classifier', inplace=True)

# Wizualizacja jakości klasyfikacji
plot_classification_quality(metrics_df)