from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

X = np.r_[np.random.randn(50, 2) + [2, 2] + np.random.randn(50, 2) + [0, -2]
          + np.random.randn(50, 2) + [-2, 2]]
print(X[:10])
print(X[51:60])
[plt.scatter(dot[0], dot[1], c='black', s=7) for dot in X]
k = 3
# X[:, 0] 代表取x X[:, 1]代表取y
C_x = np.random.randint(np.min(X[:, 0]), np.max(X[:, 0]), size=k)
C_y = np.random.randint(np.min(X[:, 1]), np.max(X[:, 1]), size=k)

C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)
plt.scatter(C_x, C_y, marker='*', s=200, c='#C0FF44')
plt.show()


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# prepare a distance set
C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
delta = dist(C, C_old, None)


def plot_kmean(current_cluster, delta):
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    fig, ax = plt.subplots()
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if current_cluster[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#C0FF44')
    plt.title('delta will be:%.4f' % delta)
    plt.plot()
    plt.show()


# converge
while delta != 0:
    print('start a new iteration')
    # find point in which k
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    # calculate new k
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    # movement for the 'kmean'
    delta = dist(C, C_old, None)
    plot_kmean(clusters, delta)
