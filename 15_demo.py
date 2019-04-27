import sklearn.datasets as datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
data = iris.data
target = iris.target

logisticRegression1 = LogisticRegression()
score = model_selection.cross_val_score(logisticRegression1, data, target, cv=3)
print(score)
logisticRegression1.fit(data, target)
print(logisticRegression1.coef_)
print(logisticRegression1.intercept_)

print(logisticRegression1.predict_proba([[5.1, 3.5, 1.4, 0.2]]))
print(logisticRegression1.predict_proba([[4.9, 2.4, 3.3, 1.0]]))
print(logisticRegression1.predict_proba([[5.9, 3.0, 5.1, 1.8]]))
print(logisticRegression1.predict([[5.1, 3.5, 1.4, 0.2]]))
print(logisticRegression1.predict([[4.9, 2.4, 3.3, 1.0]]))
print(logisticRegression1.predict([[5.9, 3.0, 5.1, 1.8]]))
