# %%
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

# %%
file_path = r'C:\Users\tiara.santoso\Downloads\20240318-result_beauty_persona.xlsx'
sheet_name1 = 'train'
sheet_name2 = 'test'
df_train = pd.read_excel(file_path, sheet_name=sheet_name1)
df_test = pd.read_excel(file_path, sheet_name=sheet_name2)


df_test.head()


# %%
df_train.info()

# %%
from sklearn.tree import DecisionTreeClassifier

X_train = df_train.drop(columns=['Cluster','person_id'])
y_train = df_train['Cluster']

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

X_test = df_test.drop(columns=['person_id'])
predicted_clusters = decision_tree_model.predict(X_test)

df_test['predicted_cluster'] = predicted_clusters

# %%
df_test.head()

# %%
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Silhouette Score
silhouette = silhouette_score(X_test, predicted_clusters)
print("Silhouette Score:", silhouette)

# Davies-Bouldin Index
db_index = davies_bouldin_score(X_test, predicted_clusters)
print("Davies-Bouldin Index:", db_index)

# Calinski-Harabasz Index
ch_index = calinski_harabasz_score(X_test, predicted_clusters)
print("Calinski-Harabasz Index:", ch_index)


# %%
df_test.head()

# %%
from sklearn.manifold import TSNE

# Reduce the dimensionality of the data to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(X_test)

# Plotting clusters using t-SNE
plt.figure(figsize=(8, 6))
for cluster_label in range(5): 
    cluster_points = tsne_data[df_test['predicted_cluster'] == cluster_label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}', alpha=0.5)

    # Compute centroid of the cluster
    centroid = np.mean(cluster_points, axis=0)
    
    # Connect data points to centroid
    for point in cluster_points:
        plt.plot([centroid[0], point[0]], [centroid[1], point[1]], linestyle='-', color='gray', alpha=0.3)

    # Plot centroid
    plt.plot(centroid[0], centroid[1], marker='o', markersize=8, color='black')

plt.title('KMeans Clustering with t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.grid(True)
plt.show()

# %%
df_test.to_excel('20240320-result_beauty_persona_booster.xlsx',index=True)

# %%


# %%
from sklearn.preprocessing import StandardScaler

df.set_index('person_id', inplace=True)

scaler = StandardScaler()
# df_scaled = df.drop(columns=['person_id'])
scaled_data = scaler.fit_transform(df)


# %%
print(scaled_data)

# %%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# %%
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(scaled_data)

df['Cluster'] = kmeans.labels_

# %%
df.head()

# %%

plt.figure(figsize=(8, 6))
for cluster_label in range(5): 
    cluster_points = scaled_data[df['Cluster'] == cluster_label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}', alpha=0.5)

    # Compute centroid of the cluster
    centroid = np.mean(cluster_points, axis=0)
    
    # Connect data points to centroid
    for point in cluster_points:
        plt.plot([centroid[0], point[0]], [centroid[1], point[1]], linestyle='-', color='gray', alpha=0.3)

    # Plot centroid
    plt.plot(centroid[0], centroid[1], marker='o', markersize=8, color='black')

plt.title('KMeans Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# %%
from sklearn.manifold import TSNE

# Reduce the dimensionality of the data to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(scaled_data)

# Plotting clusters using t-SNE
plt.figure(figsize=(8, 6))
for cluster_label in range(5): 
    cluster_points = tsne_data[df['Cluster'] == cluster_label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}', alpha=0.5)

    # Compute centroid of the cluster
    centroid = np.mean(cluster_points, axis=0)
    
    # Connect data points to centroid
    for point in cluster_points:
        plt.plot([centroid[0], point[0]], [centroid[1], point[1]], linestyle='-', color='gray', alpha=0.3)

    # Plot centroid
    plt.plot(centroid[0], centroid[1], marker='o', markersize=8, color='black')

plt.title('KMeans Clustering with t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.grid(True)
plt.show()

# %%
df.head()

# %%
df.to_excel('20240320-result_beauty_persona_main_and_booster.xlsx',index=True)


