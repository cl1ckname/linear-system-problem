from typing import List, Union
from .utility import permutation

from .vector import Vector
from .matrix import Matrix, zeros

@permutation
def SeidelSolve(A: Matrix, b: Union[Vector, List[float]], eps: float):
    b = Vector.fromIterable(b)
    b = A.T * b
    A = A.T*A 
    assert len(A) == len(b)
    if isinstance(b, list):
        b = Vector.fromIterable(b)
    n = len(A)
    d = Vector.fromIterable([b[i] / A[i, i] for i in range(n)])
    C = zeros(n)
    for i in range(n):
        for j in range(n):
            C[i, j] = 0 if (i == j) else - (A[i, j] / A[i, i])
    x = d.copy()
    k = 0
    while (A * x - b).abs() > eps:
        k += 1
        for i in range(n):
            x[i] = sum(C[i] * x) + d[i]
    return x, k

