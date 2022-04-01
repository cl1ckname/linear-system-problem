from .vector import Vector
from .utility import BotDiagSolve, TopDiagSolve, permutation
from .matrix import Matrix, ShapeException, zeros


def LU(A: Matrix):
    '''
    LU matrix decomposition. Learn more on the wiki -> https://en.wikipedia.org/wiki/LU_decomposition

    params
    ------
    A: Matrix
        Matrix for decomposition
    
    returns matrices such that L*U = A and L is the lower diagonal matrix, U is the upper diagonal matrix
    '''
    l = zeros(A.size)
    u = Matrix(A.matrix)
    for i in range(A.size):
        for j in range(i, A.size):
            l[j,i] = u[j,i] / u[i,i]
    for k in range(1, u.size):
        for i in range(k-1, A.size):
            for j in range(i, A.size):
                l[j,i] = u[j,i] / u[i,i]
        for i in range(k, u.size):
            for j in range(k-1, u.size):
                u[i, j] = u[i, j] - l[i, k-1] * u[k-1, j]

    return l, u

@permutation
def SolveLU(A: Matrix, b: Vector):
    '''
    Solves the system `A*x = b` as follows: `L*y=b` - > we get y. `U*x = y` -> we get x.
    Time complesety - O(n^3)

    params
    ------
    A: Matrix
        matrix of system
    b: Vector
        free vector of system
    
    returns solution vector
    '''
    assert len(A) == len(b), ShapeException(len(A), len(b))
    l, u = LU(A)
    y = BotDiagSolve(l, b)
    x = TopDiagSolve(u, y)
    return x