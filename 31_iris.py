import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()

X = iris.data
species = iris.target

fig = plt.figure(1, figsize=(8, 12))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=species, cmap=plt.cm.Paired)
ax.set_xlabel("1st eigen")
ax.set_ylabel("2nd eigen")
ax.set_zlabel("3rd eigen")
plt.show()