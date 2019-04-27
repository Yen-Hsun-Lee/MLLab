from sklearn import datasets
import numpy

regressionData = datasets.make_regression(10, 6, noise=5)
regressionX = regressionData[0]
print(type(regressionX))
r1 = numpy.array(sorted(regressionX, key=lambda tup: tup[0]))
r2 = sorted(regressionX, key=lambda tup: tup[1])
r3 = sorted(regressionX, key=lambda tup: tup[2])
r4 = sorted(regressionX, key=lambda tup: tup[3])
r5 = sorted(regressionX, key=lambda tup: tup[4])
r6 = sorted(regressionX, key=lambda tup: tup[5])
print("finished")

