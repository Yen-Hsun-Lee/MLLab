from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]

clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)

print(clf.predict([[-1, -1], [0, -1],
                   [-1, 0], [2, 2], [2, 0], [0, 2]]))