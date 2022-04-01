from math import sqrt as sq
from typing import Callable, List

from .vector import Vector
from .matrix import Matrix, getP

def sqrt(x: float):
    return sq(x)

def permutation(f: Callable[[Matrix, List[float]], Vector]):
    def solution(A: Matrix, b: List[float], eps: float = 0):
        P = getP(A)
        if eps:
            x = f(P*A, P * b, eps)
        else:
            x = f(P*A, P * b)
        return x
    return solution



def BotDiagSolve(A: Matrix, b: List[float]):
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
    assert len(A) == len(b)
    n = len(A)
    x = Vector(n)
    for i in range(n-1, -1, -1):
        bi = b[i]
        for j in range(i+1, n):
            bi -= A[i, j] * x[j]
        x[i] = bi/A[i, i]
    return x