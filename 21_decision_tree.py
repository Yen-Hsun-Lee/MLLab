from subprocess import check_call

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz

X = [[0, 0], [1, 1], [0, 1], [1, 0]]
Y = [0, 0, 1, 1]
colors = ['red', 'green']
marker = ['o', '*']
index = 0
while index < len(X):
    type = Y[index]
    plt.scatter(X[index][0], X[index][1], c=colors[type], marker=marker[type])
    index += 1
plt.show()

clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)
# manual make a directory graph
export_graphviz(clf, out_file='graph\\demo21.dot', filled=True,
                rounded=True, special_characters=True)
check_call(['dot','-Tsvg', 'graph\\demo21.dot', '-o', 'graph\\demo21_output.svg'])