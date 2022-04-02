from typing import List
import numpy as np

from methods.QR import QRSolve
from methods.vector import Vector
from methods.matrix import Matrix, eye, zeros

N = 2 

class Test:
    '''
    Test class that incapsulate the matrix of system, free vector and solution vector
    '''
    A: Matrix
    b: List[float]
    x: Vector

    def __init__(self, A: List[List[float]], b: List[float]):
        '''
        Calculus solution vector
        params
        ------
        A: Matrix
            Matrix of the system
        b: List[float] | Vector
            Free vector of the system

        '''
        self.A = Matrix(A)
        self.b = b
        self.x = Vector.fromIterable(np.linalg.solve(self.A.matrix, b))


test0 = Test([
    [0, 2, 3],
    [1, 2, 4],
    [4, 5, 6]
],
    [13, 27, 32]
)

test1 = Test([
    [N+2, 1, 1],
    [1, N+4, 1],
    [1, 1, N+6]
],
    [N+4, N+6, N+8]
)

test2 = Test([
    [-(N+2), 1, 1],
    [1, -(N+4), 1],
    [1, 1, -(N+6)]
],
    [-(N+4), -(N+6), -(N+8)]
)

test3 = Test([
    [-(N+2), N+3, N+4],
    [N+5, -(N+4), N+1],
    [N+4, N+5, -(N+6)]
],
    [N+4, N+6, N+8]
)

test4 = Test([
    [N+2, N+1, N+1],
    [N+1, N+4, N+1],
    [N+1, N+1, N+6]
],
    [N+4, N+6, N+8]
)

def generateTest5(n: int, eps: float):
    '''
    Generate a system with a weakly conditioned matrix

    params
    ------
    n: int
        number of equations and variables
    eps:
        the amount of conditionality of the system
    '''
    A1 = eye(n)
    for i in range(n):
        for j in range(i+1,n):
            A1[i, j] = -1
    A2 = zeros(n)
    for i in range(n):
        for j in range(n):
            A2[i, j] = -1 if j > i else 1
    A2 = A2 * (N*eps)
    A = A1 + A2
    b = [-1 for i in range(n)]
    b[-1] = 1
    return Test(A, b)

test_1 = Test([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
],
    [3, 4, 5]
)

tests = [test0, test1, test2, test3, test4]

if __name__ == '__main__':
    print(QRSolve(test2.A, test2.b))
    print(test2.x)