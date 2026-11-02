import matplotlib.pyplot as plt
from tt1 import lu, solve_system, inverse_matrix, vec_multiplication, multiplicate
import numpy as np
a = 3
eps = 0.00001


def F(x):
    f1 = (x[0] ** 2 + a ** 2) * x[1] - a ** 3
    f2 = (x[0] - a / 2) ** 2 + (x[1] - a / 2) ** 2 - a ** 2
    return [f1, f2]


def jacobian(x):
    J = [[0, 0], [0, 0]]
    J[0][0] = 2 * x[0] * x[1]
    J[0][1] = x[0] ** 2 + a ** 2
    J[1][0] = 2 * (x[0] - a / 2)
    J[1][1] = 2 * (x[1] - a / 2)
    return np.array(J)


def norm(v):
    m = 0
    for i in v:
        if abs(i) > m:
            m = abs(i)
    return m


def matrix_norm(A):
    m = 0
    for row in A:
        s = 0
        for value in row:
            s = s + abs(value)
        if s > m:
            m = s
    return m


def difference(u, v):
    d = []
    for i in range(len(u)):
        d.append(u[i] - v[i])
    return d


def phi_norm(x):
    LJ = multiplicate(L, jacobian(x))
    for i in range(2):
        for j in range(2):
            if i == j:
                LJ[i][j] = 1 - LJ[i][j]
            else:
                LJ[i][j] = -LJ[i][j]
    return matrix_norm(LJ)


x0 = [4.5, 1.0]
L0, U0, P0, swap0 = lu(jacobian(x0))
L = inverse_matrix(L0, U0, P0)


def simple_iteration(x, eps):
    xs = [x]
    e = eps + 1
    while e > eps:
        step = vec_multiplication(L, F(x))
        x_new = [x[0] - step[0], x[1] - step[1]]
        e = q / (1 - q) * norm(difference(x_new, x))
        x = x_new
        xs.append(x)
    return xs


def newton(x, eps):
    xs = [x]
    e = eps + 1
    while e > eps:
        fx = F(x)
        minus_f = [-fx[0], -fx[1]]
        Lm, U, P, swap = lu(jacobian(x))
        dx = solve_system(Lm, U, P, minus_f)
        x = [x[0] + dx[0], x[1] + dx[1]]
        e = norm(dx)
        xs.append(x)
    return xs


x_star = newton(x0, 1e-14)[-1]
middle = [(x0[0] + x_star[0]) / 2, (x0[1] + x_star[1]) / 2]

q = phi_norm(x0)
if phi_norm(middle) > q:
    q = phi_norm(middle)
if phi_norm(x_star) > q:
    q = phi_norm(x_star)

si = simple_iteration(x0, eps)
nt = newton(x0, eps)
print("простая итерация", si[-1], "итераций", len(si) - 1)
print("ньютон", nt[-1], "итераций", len(nt) - 1)

grid = []
for i in range(-100, 121):
    grid.append(i * 0.05)

z1 = []
z2 = []
for v in grid:
    row1 = []
    row2 = []
    for u in grid:
        values = F([u, v])
        row1.append(values[0])
        row2.append(values[1])
    z1.append(row1)
    z2.append(row2)

plt.contour(grid, grid, z1, levels=[0], colors="blue")
plt.contour(grid, grid, z2, levels=[0], colors="red")
plt.plot([x0[0]], [x0[1]], "ko")
plt.grid(True)
plt.show()

si = simple_iteration(x0, 1e-14)
nt = newton(x0, 1e-14)

err_si = []
for x in si:
    err_si.append(norm(difference(x, x_star)))

err_nt = []
for x in nt:
    err_nt.append(norm(difference(x, x_star)))

plt.semilogy(err_si, "o-", label="простая итерация")
plt.semilogy(err_nt, "s-", label="ньютон")
plt.xlabel("k")
plt.ylabel("погрешность")
plt.legend()
plt.grid(True)
plt.show()
