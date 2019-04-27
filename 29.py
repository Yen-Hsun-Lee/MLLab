import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1, ], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[-8, -2], [-0.8, 2], [4, -5]]))

clf_pf = GaussianNB()
# 限制np.unique(Y)  調味
clf_pf.partial_fit(X, Y, np.unique(Y))
print(clf_pf.predict([[-0.8, -1]]))
clf_pf.partial_fit([[4, 4], [5, 5], [-2, -2]], [1, 1, 2])
print(clf_pf.predict([[0.8, 1]]))