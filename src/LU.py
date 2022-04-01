from .vector import Vector
from .utility import BotDiagSolve, TopDiagSolve, permutation
from .matrix import Matrix, ShapeException, zeros


def LU(a: Matrix):
    l = zeros(a.size)
    u = Matrix(a.matrix)
    for i in range(a.size):
        for j in range(i, a.size):
            l[j,i] = u[j,i] / u[i,i]
    for k in range(1, u.size):
        for i in range(k-1, a.size):
            for j in range(i, a.size):
                l[j,i] = u[j,i] / u[i,i]
        for i in range(k, u.size):
            for j in range(k-1, u.size):
                u[i, j] = u[i, j] - l[i, k-1] * u[k-1, j]

    return l, u

@permutation
def SolveLU(A: Matrix, b: Vector):
    assert len(A) == len(b), ShapeException(len(A), len(b))
    l, u = LU(A)
    y = BotDiagSolve(l, b)
    x = TopDiagSolve(u, y)
    return x