import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import numpy as np

np.random.seed(20190413)  # only when you want the result is REPEATABLE

regressionData = datasets.make_regression(100, 1, noise=10)
print(type(regressionData), len(regressionData))
print(type(regressionData[0]), type(regressionData[1]))
print(regressionData[0].shape, regressionData[1].shape)
plt.scatter(regressionData[0], regressionData[1], c='red', marker='^')
# plt.axis([-5, 5, -150, 150])
# plt.show()

regression1 = linear_model.LinearRegression()
regression1.fit(regressionData[0], regressionData[1])
print('coef, intercept', regression1.coef_, regression1.intercept_)
print('score:', regression1.score(regressionData[0], regressionData[1]))

range1 = [-3, 3]
plt.plot(range1, regression1.coef_ * range1, regression1.intercept_)
plt.show()

import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import numpy as np
np.random.seed(20190413)
regData = datasets.make_regression(100)