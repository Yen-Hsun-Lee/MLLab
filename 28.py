import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

df1 = pd.read_csv('data\\sonar.all-data', header=None, prefix='X')
print(df1.shape)
print(type(df1), df1.iloc[:, 60].unique())
print(df1.head())
data, labels = df1.ix[:, :-1], df1.ix[:, -1]
print(data.shape)
print(labels.shape)


df1.rename(columns={'X60': 'Label'}, inplace=True)
# n_neighbors關聯數 2 為自己跟最近的一個 3 為自己關聯最近的一個在關聯最近的
clf = KNeighborsClassifier(n_neighbors=3)

X_train, X_test, Y_Train, Y_Test = train_test_split(data, labels, test_size=0.2)
clf.fit(X_train, Y_Train)
y_predict = clf.predict(X_test)
print('score=',clf.score(X_test, Y_Test))

result_cm1 = confusion_matrix(Y_Test, y_predict)
print(result_cm1)

scores = cross_val_score(clf, data, labels, cv=5, groups=labels)
print(scores)