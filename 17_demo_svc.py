import numpy as np
from sklearn.svm import SVC

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
classifier = SVC()
classifier.fit(X, y)
print("predict 4,4", classifier.predict([[4, 4]]))
print("predict 4,-4", classifier.predict([[4, -4]]))
print("predict -4,-4", classifier.predict([[-4, -4]]))
print("predict -4,4", classifier.predict([[-4, 4]]))