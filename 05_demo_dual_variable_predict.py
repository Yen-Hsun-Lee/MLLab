from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [1, 4, 8])

print("coef", reg.coef_)
print("intercept", reg.intercept_)

print(reg.predict([[0.8, 0.8], [2, 1], [10, 14]]))
result = reg.predict([[0.8, 0.8], [2, 1], [10, 14]])

print(reg.score([[0, 0], [1, 1], [2, 2]], [1, 4, 8]))
print(reg.score([[0, 0], [1, 1], [2, 2]], [1, 3, 9]))


# from sklearn import  linear_model
# reg = linear_model.LinearRegression()
# reg.fit(())