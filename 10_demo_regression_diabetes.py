import numpy as np
from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
print("type=", type(diabetes))
print("feature shape=", diabetes.data.shape)
print("label shape=", diabetes.target.shape)
# print("first few data:", diabetes.data[:10])
# print("first few target:", diabetes.target[:10])

# 切割資料
dataForTest = -50
data_train = diabetes.data[:dataForTest]
# print("model train with data shape:",data_train.shape)
target_train = diabetes.target[:dataForTest]
# print("model train with target shape:",target_train.shape)

data_test = diabetes.data[dataForTest:]
# print("model test with data shape:", data_test.shape)
target_test = diabetes.target[dataForTest:]
# print("model test with target shape:", target_test.shape)

# build model, fit data
regression1 = linear_model.LinearRegression()
regression1.fit(data_train, target_train)

# check model
# print(regression1.coef_)
# print(regression1.intercept_)

print(regression1.score(data_test, target_test))
# print(data_test)
for i in range(dataForTest, 0):
    # print("original, ith data shape:",data_test[i].shape)
    # print('using a list to wrap, shape:',data_test[i].reshape(1,-1).shape)
    print('predict = %.2f, actual = %.2f' % (regression1.predict(data_test[i].reshape(1, -1))[0]
                                        , target_test[i]))