# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Step 2: Load the data
data = pd.read_csv('autos.csv')
df = data[['horsepower', 'price']]
# Step 3: Preprocess the data
print("data: ", df)
# Convert categorical data to numerical data
le = LabelEncoder()
# for i in df.columns:
#     if df[i].dtype == 'object':
#         df[i] = le.fit_transform(df[i].astype(str))

# Handle missing values
df = df.fillna(0)

# Normalize the data
scaler = MinMaxScaler()
# df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Step 4: Choose the number of clusters (K) for K-means algorithm
# In this example, let's assume we want 3 clusters
k = 3

# Step 5: Apply the K-means algorithm
print(df)
kmeans = KMeans(n_clusters=k)
# kmeans.fit(df)
y_pred = kmeans.fit_predict(df)
# Step 6: Evaluate the model
# Print the cluster centers
print(kmeans.cluster_centers_)

df['Cluster'] = y_pred
# print(df.to_string())
data1 = df[df['Cluster'] == 0]
data2 = df[df['Cluster'] == 1]
data3 = df[df['Cluster'] == 2]
plt.scatter(data1[['horsepower']], data1[['price']], color='red')
plt.scatter(data2[['horsepower']], data2[['price']], color='orange')
plt.scatter(data3[['horsepower']], data3[['price']], color='green')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],c='pink',edgecolors='black',marker='*',s = 100)

plt.show()