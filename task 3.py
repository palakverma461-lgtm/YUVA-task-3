# =========================================================
# YUVA INTERNSHIP - DATA SCIENCE
# Week 3 Task: Unsupervised Learning and Clustering Analysis
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

print("--- Starting Week 3 Task: K-Means Clustering ---")

# 1. Dataset Selection & Generation
# We are generating a synthetic Customer Dataset with 'Annual Income' and 'Spending Score'
# This is a standard approach to demonstrate customer segmentation.
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
df = pd.DataFrame(X, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
df['Annual Income (k$)'] = df['Annual Income (k$)'] * 10 + 60
df['Spending Score (1-100)'] = df['Spending Score (1-100)'] * 10 + 50

print("Dataset Created. Shape:", df.shape)

# 2. Preprocessing
# Standardizing the data is crucial for distance-based algorithms like K-Means
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 3. Finding the Optimal Number of Clusters (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plotting Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('The Elbow Method to find Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig('Elbow_Method.png', bbox_inches='tight') # Save figure for DOCX
plt.show()

print("From the Elbow plot, the optimal number of clusters is chosen as K=4.")

# 4. Applying K-Means Clustering
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# 5. Visualizing the Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster', data=df, palette='viridis', s=100)
plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.savefig('Customer_Clusters.png', bbox_inches='tight') # Save figure for DOCX
plt.show()

print("\n--- Clustering Completed. Graphs saved as PNG files! ---")
# Now, upload this code to GitHub and paste the PNGs in your DOCX report.