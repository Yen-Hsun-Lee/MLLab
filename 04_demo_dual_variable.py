# import matplotlib.pyplot as plt
# from sklearn import linear_model
#
# features = [[0, 1], [1, 3], [2, 8]]
# values = [1, 4, 5.5]
# reg = linear_model.LinearRegression()
# reg.fit(features, values)
# print('coefficient', reg.coef_)
# print('intercept', reg.intercept_)
# print(reg.coef_[0], reg.coef_[1])
#
# plt.scatter([[0], [1], [2]], [1, 4, 5.5], c='green')
# plt.scatter([[1],[3],[8]],[1,4,5.5], c='blue')
# plt.show()

import matplotlib.pyplot as plt
from sklearn import linear_model
features = [[0,1], [1,3],[2,8]]
values=[1,4,5.5]
reg = linear_model.LinearRegression()
reg.fit(features, values)
print('coefficient', reg.coef_)
print('intercept', reg.intercept_)
print(reg.coef_[0], reg.coef_[1])

plt.scatter([[0], [1], [2]], [1,4,5.5], c='green')
plt.scatter([[1], [3],[9]])