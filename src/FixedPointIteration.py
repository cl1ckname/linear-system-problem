from typing import List

from .vector import Vector
from .matrix import Matrix, eye


def IterationSolve(A: Matrix, b: List[float], eps: float):
    '''
    Fixed point iterative method for solving systems of algebraic 
    linear equations. Learn more -> https://en.wikipedia.org/wiki/Fixed-point_iteration

    params
    ------
    A: Matrix
        Matrix of the system
    b: Vactor | List[float]
        Free vector of the system
    eps: float
        Required methodological error of the solution

    returns
    -------
    x: Vector
        Vector of solution
    k: int
        Number of iterations
    '''
    n = len(A)
    bv = Vector.fromIterable(b)
    for i in range(4):
        abs_a = A.abs(i)
        mu = 1 / abs_a
        B = eye(n) - A * mu
        abs_b = B.abs(i)
        if abs_b < 1:
            break
    else:
        bv = A.T * bv
        A = A.T * A
        for i in range(4):
            abs_a = A.abs()
            mu = 1 / abs_a
            B = eye(n) - A * mu
            abs_b = B.abs(i)
            if abs_b < 1:
                break
        else:
            raise Exception('Norm not found...')

    assert abs_b < 1, abs_b
    c = Vector.fromIterable([mu * i for i in bv])
    x = c.copy()
    k = 0
    while (A * x - bv).abs() > eps:
        k += 1
        x = B * x + c
    return x, k