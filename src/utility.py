from typing import Callable, List

from .vector import Vector
from .matrix import Matrix, getP

def permutation(f: Callable[[Matrix, List[float]], Vector]):
    '''
    A decorator that multiplies the original matrix 
    and the free vector by the transformation matrix 
    to avoid zeros on the diagonal of the matrix of the 
    system. The solution of the new system coincides 
    with the solution of the original one
    '''
    def solution(A: Matrix, b: List[float], eps: float = 0):
        P = getP(A)
        if eps:
            x = f(P*A, P * b, eps)
        else:
            x = f(P*A, P * b)
        return x
    return solution



def BotDiagSolve(A: Matrix, b: List[float]):
    '''
    Solution of a system with a lower diagonal matrix by the Gauss method.

    params
    ------
    A: Matrix
        Matrix of linear system
    b: Vector | List[float]
        free vector of system
    
    returns the solution vector
    '''
    assert len(A) == len(b)
    n = len(A)
    x = Vector(n)
    for i in range(n):
        bi = b[i]
        for j in range(0,i):
            bi -= A[i, j] * x[j]
        x[i] = bi/A[i, i]
    return x

def TopDiagSolve(A: Matrix, b: List[float]):
    '''
    Solution of a system with a top diagonal matrix by the Gauss method.

    params
    ------
    A: Matrix
        Matrix of linear system
    b: Vector | List[float]
        free vector of system
    
    returns the solution vector
    '''
    assert len(A) == len(b)
    n = len(A)
    x = Vector(n)
    for i in range(n-1, -1, -1):
        bi = b[i]
        for j in range(i+1, n):
            bi -= A[i, j] * x[j]
        x[i] = bi/A[i, i]
    return x