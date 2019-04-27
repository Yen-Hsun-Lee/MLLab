import matplotlib.pyplot as plt
import numpy as np

from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1, ], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

x_min = -4
x_max = 4
y_min = -4
y_max = 4

h = .025
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), (y_min, y_max, h))
print(xx.shape, yy.shape)
clf = GaussianNB()
clf.fit(X, Y)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.pcolormesh(xx, yy, Z)
plt.show()
