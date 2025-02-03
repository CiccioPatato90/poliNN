import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from db import Database
from sklearn.preprocessing import StandardScaler

if False:
    db = Database('res/records.db')
    data = db.fetch_all()
    data = pd.DataFrame(data)
else:
    data =pd.read_csv('res/processed_data.csv')
# View the first few rows
print(data.head())

# Get a summary of the data
print(data.info())

# Statistical summary
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Drop or fill missing values
# Option 1: Drop missing values
data = data.dropna()

X = data


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow.png')

k_optimal = 4  # Replace with your optimal number
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

data['LABEL'] = cluster_labels

data.to_csv('res/8_cluster.csv', index=False)





