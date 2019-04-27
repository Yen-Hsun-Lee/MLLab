import math


def square_root(x):
    return math.sqrt(x)


print(square_root(100))


def square_root_lambda(x): return math.sqrt(x)


print(square_root_lambda(100))

square_root_lambda2 = lambda x: math.sqrt(x)

print(square_root(5), square_root_lambda(5), square_root_lambda2(5))