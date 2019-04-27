import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

shortestNeighbors = NearestNeighbors(n_neighbors=2).fit(X)
distances, indices = shortestNeighbors.kneighbors(X, return_distance=True)
print("distances",distances)
print("indexes", indices)