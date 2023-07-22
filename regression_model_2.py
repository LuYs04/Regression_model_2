import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from random import randint
import math

exp_count = int(input("Experiments' count: "))
x_len = int(input("Count of x-es: "))


# input X
def inputX(n, x_len):
    x = np.zeros((n, x_len))
    for i in range(n):
        for j in range(x_len):
            # x[i][j] = eval(input("x = "))
            x[i][j] = randint(1, 7)
    return (x)


# input Y
def inputY(n):
    y = np.zeros(n)
    for i in range(n):
        # y[i] = eval(input("y = "))
        y[i] = randint(1, 7)
    return (y)


# y_average
def avg_y(y, n):
    avg = 0
    for i in y:
        avg += i
    return (avg / n)


# x_average
def avg_xj(x, n, j):
    avg = 0
    for i in range(n):
        avg += x[i][j]
    return (avg / n)


# sigma_j
def sigmaj(x, n, j):
    s = 0
    for i in range(n):
        s += (x[i][j] - avg_xj(x, n, j)) ** 2
    return (math.sqrt(s / (n - 1)))


# sigma_0
def sigma0(y, n):
    s = 0
    for i in range(n):
        s += (y[i] - avg_y(y, n)) ** 2
    return (math.sqrt(s / (n - 1)))


# getting X*
def gettingX(x, n, x_len):
    x_n = np.zeros((n, x_len))
    for j in range(x_len):
        for i in range(n):
            x_n[i][j] = (x[i][j] - avg_xj(x, n, j)) / sigmaj(x, n, j)

    return (x_n)


# getting Y*
def gettingY(y, n):
    y_n = np.zeros(n)
    for i in range(n):
        y_n[i] = (y[i] - avg_y(y, n)) / sigma0(y, n)
    return (y_n)


# sum_x
def sum_x(x, i, n):
    s = 0
    for k in range(n):
        s += x[k, i]
    return (s)


# sum_xx
def sum_xx(x, i, j, n):
    s = 0
    for k in range(n):
        s += x[k, i] * x[k, j]
    return (s)


# getting R
def gettingR(x_n, n, k):
    r = np.zeros((k, k))
    for j in range(k):
        for m in range(k):
            if j == m:
                r[j][m] = 1
            else:
                for i in range(n):
                    r[j][m] += (x_n[i][j] * x_n[i][m]) / (n - 1)
    return r


# getting r0
def gettingR0(k, n, y_n, x_n):
    r0 = np.zeros(k)
    for j in range(k):
        for i in range(n):
            r0[j] += (y_n[i] * x_n[i][j]) / (n - 1)
    return (r0)


# getting A*
def getting_a(R, R0):
    if det(R) != 0:
        r_1 = inv(R)
    else:
        print("the determinant of R is 0 :(")
    # print("A_inv = \n", g, "\n\n")
    a = np.matmul(r_1, R0)
    return (a)


# ss_o
def ss_o(Y, avg_y):
    ss_o = 0
    for i in Y:
        ss_o += (i - avg_y) ** 2
    return (ss_o)


# ss_r
def ss_r(Y_m, avg_y):
    ss_r = 0
    for i in Y_m:
        ss_r += (i - avg_y) ** 2
    return (ss_r)


# ss_e
def ss_e(Y, Y_m):
    ss_e = 0
    for i in range(len(Y)):
        ss_e += (Y[i] - Y_m[i]) ** 2
    return (ss_e)


# getting Y_m
def gettingY_m(n, x_len, b, x):
    y_m = np.zeros(n)
    for i in range(n):
        for k in range(x_len + 1):
            if k == 0:
                y_m[i] += b[k]
            else:
                y_m[i] += b[k] * x[i, k - 1]
    return (y_m)


# getting b*
def getting_b(a, k, n, x, y):
    b = np.zeros(k + 1)
    for j in range(1, k + 1):
        b[j] = a[j - 1] * sigma0(y, n) / sigmaj(x, n, j - 1)
        b[0] += b[j] * avg_xj(x, n, j - 1)
    b[0] = avg_y(y, n) - b[0]
    return (b)


# main
x = np.array(
    [[3, 6, 7, 7], [7, 4, 4, 2], [4, 3, 1, 1], [6, 3, 6, 5], [3, 1, 7, 2], [6, 1, 6, 6], [6, 2, 6, 4], [4, 4, 2, 5],
     [5, 2, 2, 7], [1, 7, 7, 5], [5, 1, 1, 4], [4, 1, 2, 4], [5, 7, 2, 3], [7, 5, 7, 3], [4, 3, 1, 5]])
y = np.array([4, 7, 2, 7, 6, 2, 4, 5, 4, 3, 5, 2, 3, 1, 6])
# x = inputX(exp_count, x_len)
# y = inputY(exp_count)
y_n = gettingY(y, exp_count)
x_n = gettingX(x, exp_count, x_len)
R = gettingR(x_n, exp_count, x_len)
R0 = gettingR0(x_len, exp_count, y_n, x_n)
a = getting_a(R, R0)
b = getting_b(a, x_len, exp_count, x, y)
y_m = gettingY_m(exp_count, x_len, b, x)
ss0 = ss_o(y, avg_y(y, exp_count))
ssE = ss_e(y, y_m)
ssR = ss_r(y_m, avg_y(y, exp_count))
r_2 = ssR / ss0
msR = ssR / x_len
sigma_2 = ssE / (exp_count - x_len - 1)
F = msR / sigma_2

print("X = \n", x, "\n\n")
print("Y = \n", y, "\n\n")
print("Y_n = \n", y_n, "\n\n")
print("X_n = \n", x_n, "\n\n")
print("R = \n", R, "\n\n")
print("R0 = \n", R0, "\n\n")
print("a = \n", a, "\n\n")
print("b = \n", b, "\n\n")
print("Y_m = \n", y_m, "\n\n")
print("ss(0) = ", ss0, "\n\n")
print("ss(E) = ", ssE, "\n\n")
print("r^2 = ", r_2, "\n\n")
print("msR = ", msR, "\n\n")
print("sigma^2 = ", sigma_2, "\n\n")
print("F = ", F, "\n\n")

'''if (int(ss0) == int(ssE) + int(ssR)):
    print("Yeah! You did it!\nYour model is equivalent!!")
else:
    print("O_o! Your model isn't equivalent :(")
'''

