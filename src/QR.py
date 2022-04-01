from functools import reduce
from typing import List

from .utility import TopDiagSolve, permutation
from .matrix import Matrix, dot, eye
from .vector import ort


def QR(A: Matrix):
    '''
    Decomposes the original matrix an orthogonal matrix Q and an upper triangular matrix R. 
    Learn more - https://en.wikipedia.org/wiki/QR_decomposition .

    params
    ------
    A: Matrix
        Matrix for decomposition
    
    returns matrices such that L*U = A and L is the lower diagonal matrix, U is the upper diagonal matrix
    '''
    n = len(A)
    Q = eye(n)
    Q_steps: List[Matrix] = []
    R = Matrix(A.matrix)
    for i in range(n-1):
        Rc = R[i:]
        y = Rc[:,0]
        a = y.abs()
        z = ort(n-i, 0)
        w = y - z * a
        w = w / w.abs()
        Q_ii = eye(n-i) - dot(w,w) * 2
        R_ii = Q_ii * Rc
        for x in range(n-i):
            for y in range(n-i):
                R[i+x,i+y] = R_ii[x,y]
        Q = Q_ii.extend(n)
        Q_steps.append(Q)

    Q = reduce(lambda x,y: y*x, reversed(Q_steps))
    return Q, R

@permutation
def QRSolve(A: Matrix, b: List[float]):
    '''
    Solves linear system with QR decomposition

    `y = Q.T * b` -> y vector. Such as the R is upper triangular matrix
    we can easy solve it.

    params
    ------
    A: Matrix
        Matrix of the system
    b: Vector | List[float]
        Free vector of the system
    '''
    n = len(A)
    assert n == len(b)
    Q, R = QR(A)
    y = Q.T * b
    return TopDiagSolve(R, y)

if __name__ == '__main__':
    A = Matrix([
        [3, 1, 1],
        [1, 5, 1],
        [1, 1, 7]
    ])
    b = [5, 7, 9]
    print(QRSolve(A,b))