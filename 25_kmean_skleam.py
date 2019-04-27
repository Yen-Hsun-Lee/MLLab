from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 0], [0, 1], [1, 2], [1, 4], [2, 0], [4, 2], [4, 4],
              [6, 0], [4, 6], [4, 7]])
kmeans = KMeans(n_clusters=3, random_state=100).fit(X)

print("labels=", kmeans.labels_)
print("predict new points", kmeans.predict([[-1, -1], [5, 5]]))
print("kmeans centers=", kmeans.cluster_centers_)