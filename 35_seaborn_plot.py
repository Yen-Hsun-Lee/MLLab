import seaborn
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = np.array([iris.target_names[i] for i in iris.target])
seaborn.pairplot(df, hue='species')
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(df[iris.feature_names], iris.target,
                                                    test_size=0.2, stratify=iris.target)
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, Y_train)

predicted = rf.predict(X_test)
accuracy = accuracy_score(Y_test, predicted)
print("OOB score estimator:{:.3}".format(rf.oob_score_))
print("mean accuracy:%.3f"%(accuracy))

cm = pd.DataFrame(confusion_matrix(Y_test, predicted), columns=iris.target_names, index=iris.target_names)
seaborn.heatmap(cm, annot=True)
plt.show()
