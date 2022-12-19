"""
compute GD in a pure python fashion.
Function f(x1, x2) x1^2 + x2^2
"""

""" x is a list of floats """


def f(x):
    return x[0]**2 + x[1]**2


def gradf(x):
    return [2*x[0], 2*x[1]]


N = 10
x0 = [2., 2.]
eps = 0.1
# x contains the current estimate = latest gd estimation
x = x0

for j in range(0, N):
    grad = gradf(x)
    x = [x[i] - eps * grad[i] for i in (0, 1)]
    print(x, f(x))
