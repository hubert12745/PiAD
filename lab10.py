import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


X, y = make_classification(n_classes=2, n_clusters_per_class=2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.show()


classifiers = {
    "GaussianNB": GaussianNB(),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(probability=True),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

results = []
lastIter = []
for name, clf in classifiers.items():
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        start_time = time.time()
        clf.fit(X_train, y_train)
        training_time = time.time() - start_time

        start_time = time.time()
        y_pred = clf.predict(X_test)
        testing_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        results.append({
            "classifier": name,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "roc_auc": roc_auc,
            "training_time": training_time,
            "testing_time": testing_time
        })
        if i == 99:
            lastIter.append({"X_test": X_test, "y_test": y_test, "y_pred": y_pred})

X_test = lastIter[0]["X_test"]
y_test = lastIter[0]["y_test"]
y_pred = lastIter[0]["y_pred"]


fpr,tpr,thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()

df = pd.DataFrame(results)
df_grouped = df.groupby("classifier").mean()
print(df_grouped)
df_grouped.to_csv("results.csv")
scale_factor = 100

# Create a copy of the dataframe to modify the scaled values
df_scaled = df_grouped.copy()
df_scaled['training_time'] = df_scaled['training_time'] * scale_factor
df_scaled['testing_time'] = df_scaled['testing_time'] * scale_factor
metrics = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc', 'training_time', 'testing_time']
n_metrics = len(metrics)
n_classifiers = len(df_scaled.index)

bar_width = 0.15
index = np.arange(n_metrics)

fig, ax = plt.subplots(figsize=(15, 8))

for i, classifier in enumerate(df_scaled.index):
    ax.bar(index + i * bar_width, df_scaled.loc[classifier, metrics], bar_width, label=classifier)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Performance of Classifiers')
ax.set_xticks(index + bar_width * (n_classifiers - 1) / 2)
ax.set_xticklabels(metrics)
ax.legend(title='Classifier')

plt.show()
