# unsupervised_learning.py
## Explore unsupervised ML model performance for identifying flowers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


## Prepare data
df = pd.read_csv("data/iris_flowers.csv")

# Classify data into features and classes
X = df.drop(columns=["class"])  # Features
y = df["class"] # Class to predict
print(X)
print(y)


# One hot encoding for flower class (converting qualitative flower names to quantitative numbers) 
y_encoded = LabelEncoder().fit_transform(y)

# Determine optimal number of clusters using the elbow method
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# Based on the graph, k=3 clusters appears to be optimal
clusters = 3

# Perform KMeans clustering with optimal k
kmeans = KMeans(n_clusters=clusters, random_state=42)
kmeans.fit(X)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=32)
kmeans.fit(X)

# Predict clusters
clusters = kmeans.labels_

# Map clusters to original classes (based on majority voting within each cluster)
def map_clusters_to_classes(y_true, clusters):
    mapping = {}
    for cluster in set(clusters):
        true_labels = y_true[clusters == cluster]
        majority_label = pd.Series(true_labels).mode()[0]
        mapping[cluster] = majority_label
    return [mapping[cluster] for cluster in clusters]

mapped_predictions = map_clusters_to_classes(y_encoded, clusters)

# Evaluation
print("Classification Report:")
print(classification_report(y_encoded, mapped_predictions, target_names=LabelEncoder().fit(y).classes_))

## Represent clusters vs features

# Scatter plot of the clusters (Sepal Length vs Sepal Width)
plt.figure(figsize=(8, 6))
plt.scatter(df["sepal-length"], df["sepal-width"], c=clusters, cmap="viridis", s=50)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroids")
plt.title("KMeans Clustering of Iris Dataset (Sepal Features)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.grid()
plt.show()

# Scatter plot of the clusters (Petal Length vs Petal Width)
plt.figure(figsize=(8, 6))
plt.scatter(df["petal-length"], df["petal-width"], c=clusters, cmap="viridis", s=50)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 2], centroids[:, 3], c="red", marker="X", s=200, label="Centroids")
plt.title("KMeans Clustering of Iris Dataset (Petal Features)")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()
plt.grid()
plt.show()  

# Confusion Matrix
cm = confusion_matrix(y_encoded, mapped_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LabelEncoder().fit(y).classes_)
disp.plot(cmap="viridis")
plt.show()
