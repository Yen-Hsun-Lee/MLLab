import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.decomposition import PCA

iris = datasets.load_iris()
pca1 = PCA(n_components=2)
data = pca1.fit(iris.data).transform(iris.data)
originalData = iris.data
print(data.shape)
print(data[0:5, ])
print(data.max(axis=0))
print(data.min(axis=0))
print(data.max(axis=1).shape)
print(data.min(axis=1).shape)
print(data.max())
print(data.min())
dataMax = data.max(axis=0) + 1
dataMin = data.min(axis=0) - 1
n = 2000
X, Y = np.meshgrid(np.linspace(dataMin[0], dataMax[0], n),
                   np.linspace(dataMin[1], dataMax[1], n))
# do the real calssification
# default (no C), C=1, C=5, C=20
# kernel='rbf'(*), 'poly', 'linear' 'sigmoid'(x)
svc = svm.SVC(C=5, kernel='sigmoid')
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
print(np.unique(Z))
plt.contour(X, Y, Z.reshape(X.shape), levels=[0, 1, 2], colors=['r', 'g', 'b'])

for c, s in zip([0, 1, 2], ['o', '+', '^']):
    d = data[iris.target == c]
    plt.scatter(d[:, 0], d[:, 1], c='k', marker=s)

plt.show()