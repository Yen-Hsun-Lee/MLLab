from sklearn import datasets
import matplotlib.pyplot as plt
import numpy
iris = datasets.load_iris()

print(type(iris))
print(iris.data.shape)
print(iris.target.shape)
# find distinct value for target (category values) // remove duplicate value and output
print(numpy.unique(iris.target))
print(iris.data[:10])

labels = iris.feature_names
X = iris.data
species = iris.target
print('start to print something')

counter = 1
for i in range(0, 4):
    for j in range(i + 1, 4):
        plt.figure(counter, figsize=(8, 6))
        counter += 1
        xData = X[:, i]
        yData = X[:, j]
        x_min, x_max = xData.min() - 0.5, xData.max() + 0.5
        y_min, y_max = yData.min() - 0.5, yData.max() + 0.5
        plt.clf()

        plt.scatter(xData, yData, c=species, cmap=plt.cm.Paired)
        plt.xlabel(labels[i])
        plt.ylabel(labels[j])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()