from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(list(iris.keys()))
print(list(iris.feature_names))
# X = iris.data[:, 3:]
X = iris.data[:, 2:3]
# X = iris.data[:, :1]
#X = iris.data[:, 1:2]
# y = (iris.target == 0).astype(np.int)
# y = (iris.target == 1).astype(np.int)
y = (iris.target == 2).astype(np.int)
print(X.shape)
print(y)

logisticRegression1 = LogisticRegression()
logisticRegression1.fit(X, y)
print(type(logisticRegression1))
print(logisticRegression1.coef_)
print(logisticRegression1.intercept_)

X_new = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
y_probability = logisticRegression1.predict_proba(X_new)

plt.plot(X, y, "gs")
plt.plot(X_new, y_probability[:,1],"g-", label="Iris-Virginica")
plt.plot(X_new, y_probability[:,0],"r--", label="Not Iris-Virginica")
plt.legend(fontsize=14)
plt.show()