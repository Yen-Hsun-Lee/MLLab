from sklearn import datasets
from sklearn import model_selection
from sklearn import svm

iris = datasets.load_iris()

svc = svm.SVC()
scores = model_selection.cross_val_score(svc, iris.data, iris.target, cv=5)
print(scores)
print("Accuracy:", scores.mean())

