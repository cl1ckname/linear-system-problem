from typing import List

from .vector import Vector
from .matrix import Matrix, eye

def stopCreteria(abs_b: float, x: Vector, x_1: Vector, eps: float) -> bool:
    abs_x = (x - x_1).abs()
    return (abs_b/(1-abs_b))*abs_x > eps


def IterationSolve(A: Matrix, b: List[float], eps: float):
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