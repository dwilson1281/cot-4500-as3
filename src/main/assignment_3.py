import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

#1
# Use Euler's Method to generate approximation of y(t)

def function(t, y):
    return t - (y ** 2)

def eulers_method(initial_point, point_a, point_b, n):
    h = (point_b - point_a) / n
    t, w = point_a, initial_point

    for i in range(n):
        w = w + (h * (function(t, w)))
        t += h

    return w

initial_point = 1
point_a, point_b = 0, 2
n = 10
print(eulers_method(initial_point, point_a, point_b, n))
print()

# 2
# Use Runge Kutta Method to generate approximation of y(t)

def runge_kutta(initial_point, point_a, point_b, n):
    h = (point_b - point_a) / n
    t, w = point_a, initial_point

    for i in range(n):
        k1 = h * function(t, w)
        k2 = h * function(t +(h/2), w + (k1/2))
        k3 = h * function(t + (h/2), w + (k2/2))
        k4 = h * function(t + h, w + k3)

        w = w + ((k1 + (2*k2) + (2*k3) +k4) / 6)
        t += h
    return w

initial_point = 1
point_a, point_b = 0, 2
n = 10
print(runge_kutta(initial_point, point_a, point_b, n))
print()

# 3
# Use Gaussian elimination and backward substitution solve the following linear system of
# equations written in augmented matrix format.

def gauss_jordan(matrix):
    length = len(matrix)
    list = []

    for i in range(length):
        max_row = i
        for j in range(i + 1, length):
            if abs(matrix[j][i]) > abs(matrix[max_row][i]):
                max_row = j

        matrix[[i, max_row]] = matrix[[max_row, i]]

        pivot = matrix[i][i]
        for j in range(i, length + 1):
            matrix[i][j] /= pivot

        for j in range(i + 1, length):
            factor = matrix[j, i]
            for k in range(length + 1):
                matrix[j][k] -= (factor * matrix[i][k])

    for i in range(length - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            factor = matrix[j, i]
            for k in range(length + 1):
                matrix[j][k] -= (factor * matrix[i][k])

    for i in range(length):
        list.append(int(matrix[i][length]))

    return list

matrix = np.array([[2., -1., 1., 6.],
                   [1., 3., 1., 0.],
                   [-1., 5., 4., -3.]])
print(np.array(gauss_jordan(matrix)))
print()

# 4
# Implement LU Factorization for the following matrix and do the following:

Matrix_q4 = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]])

def lu_decomposition(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i][i] = 1

        for j in range(i, n):
            U[i][j] = matrix[i][j]

            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]

        for j in range(i + 1, n):
            L[j][i] = matrix[j][i]

            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i]

            L[j][i] /= U[i][i]

    return L, U

# 5
# Determine if the following matrix is diagonally dominate.

def determinant(matrix):
    L, U = lu_decomposition(matrix)
    det = np.prod(np.diag(U))
    return det


det = determinant(Matrix_q4)
L, U = lu_decomposition(Matrix_q4)

print(det)
print()
print(L)
print()
print(U)
print()


def diagonally_dominant(m, n):
    for i in range(0, n):
        sum = 0

        for j in range(0, n):
            sum = sum + abs(m[i][j])

        sum = sum - abs(m[i][i])

        if (abs(m[i][i]) < sum):
            return False

    return True


n = 5

m = [[9, 0, 5, 2, 1],
     [3, 9, 1, 2, 2],
     [0, 1, 7, 2, 3],
     [4, 2, 3, 12, 2],
     [3, 2, 4, 0, 8]]

if ((diagonally_dominant(m, n))):
    print("True")
else:
    print("False")

print()

# 6
# Determine if the matrix is a positive definite.

pos_matrix = np.array([[2, 2, 1],
                   [2, 3, 0],
                   [1, 0, 2]])


def positive_definitive(pos_matrix):

    eigenvalues = np.linalg.eigvals(pos_matrix)

    for val in eigenvalues:
        if val <= 0:
            return False

    return True

print(positive_definitive(pos_matrix))