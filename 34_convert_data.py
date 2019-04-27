from numpy import array, cov, mean
from numpy.linalg import eig

A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
print(A)

M = mean(A.T, axis=1)
print(M)
M2 = mean(A.T)
print(M2)
M3 = mean(A, axis=1)
print(M3)
C = A-M
print(C)


# Covariance 共變異數
V = cov(C.T)
print(V)


# eigen 特徵值
values, vectors = eig(V)
print("vectors\n", vectors)
print("values\n", values)

P = vectors.T.dot(C.T)
print("project:\n", P.T)