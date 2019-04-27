import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import random

# for PSL python standard library
#random.seed(427)


np.random.seed(427)

X = np.r_[np.random.randn(50, 2) + [2, 2],
          np.random.randn(50, 2) + [2, -2],
          np.random.randn(50, 2) + [-2, -2],
          np.random.randn(50, 2) + [-2, 2]]
print(X.shape)
print(X[0:5, ], X[50:55], X[100:105, ], X[150:155, ])

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

colors = ['c','m','y','k']
markers = ['o','v','^','s']

for i in range(3):
    dataX = X[kmeans.labels_==i]
    plt.scatter(dataX[:,0],dataX[:,1],c=colors[i], marker=markers[i])
    print(dataX.size)
plt.show()


# https://zh.wikipedia.org/wiki/%E6%A2%85%E6%A3%AE%E6%97%8B%E8%BD%AC%E7%AE%97%E6%B3%95