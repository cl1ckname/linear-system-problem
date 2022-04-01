from typing import List, Union, overload
from copy import deepcopy

from .vector import Vector

class ShapeException(Exception):
    def __init__(self, shape1, shape2):
        self.message = f'shapes incompatible ({shape1}, {shape2})'
    def __str__(self):
        return self.message

class Matrix:
    def __init__(self, i: List[List[float]]):
        assert len(i)
        assert len(i) == len(i[0])
        self.matrix = deepcopy(i)
        self.size = len(i)

    def abs(self, t: int = 0) -> float:
        if t == 0:
            s = 0.
            for i in self.matrix:
                for j in i:
                    if abs(j) > s:
                        s = abs(j)
            return s
        elif t == 1:
            s = 0
            for i in self.matrix:
                sj = 0
                for j in i:
                    sj += abs(j)
                if sj > s:
                    s = sj
            return s
        elif t == 2:
            s = 0
            for i in self.T:
                sj = 0
                for j in i:
                    sj += abs(j)
                if sj > s:
                    s = sj
            return s
        elif t == 3:
            s = 0
            for i in self.matrix:
                for j in i:
                    s += abs(j)
            return s ** 0.5

    def swap(self, i, j):
        self.matrix[i], self.matrix[j] = self.matrix[j], self.matrix[i]

    def extend(self, n: int) -> 'Matrix':
        d = n - self.size
        assert d >= 0, 'n less then matrix size!'
        new_m = eye(n)
        for i in range(n):
            if i < d:
                continue
            for j in range(n):
                if j < d:
                    continue
                new_m[i,j] = self.matrix[i-d][j-d]
        return new_m

    @property
    def T(self):
        return Matrix([[self.matrix[i][j] for i in range(self.size)] for j in range(self.size)])

    @overload
    def __getitem__(self, pos: int) -> Vector: ...

    @overload
    def __getitem__(self, pos: tuple[int, int]) -> float: ...

    @overload
    def __getitem__(self, pos: slice) -> 'Matrix': ...

    @overload
    def __getitem__(self, pos: tuple[slice, int]) -> Vector: ...

    def __getitem__(self, pos: Union[tuple[int, int], int]):
        if isinstance(pos, tuple):
            i, j = pos
            if isinstance(i, slice):
                v = Vector(self.size)
                for i in range(self.size):
                    v[i] = self.matrix[i][j]
                return v
            else:
                assert i < self.size
                assert j < self.size
                return self.matrix[i][j]
        elif isinstance(pos, slice):
            vecs = self.matrix[pos]
            for i in range(len(vecs)):
                vecs[i] = vecs[i][pos]
            return Matrix(vecs)
        else:
            return Vector.fromIterable(self.matrix[pos])
    
    def __setitem__(self, pos: tuple[int, int], v: float):
        i,j = pos
        assert i < self.size
        assert j < self.size
        self.matrix[i][j] = v

    def __sub__(self, other: 'Matrix'):
        assert self.size == other.size, ShapeException(self.size, other.size)
        r = zeros(self.size)
        for i in range(self.size):
            for j in range(self.size):
                r[i,j] = self.matrix[i][j] - other[i, j]
        return r
    
    def __add__(self, other: 'Matrix'):
        assert self.size == other.size, ShapeException(self.size, other.size)
        r = zeros(self.size)
        for i in range(self.size):
            for j in range(self.size):
                r[i,j] = self.matrix[i][j] + other[i, j]
        return r

    @overload
    def __mul__(self, other: 'Matrix') -> 'Matrix': ...

    @overload
    def __mul__(self, other: float) -> 'Matrix': ...

    @overload
    def __mul__(self, other: 'Vector') -> 'Vector': ...

    def __mul__(self, other: 'Matrix | float | Vector'):
        r = zeros(self.size)
        if isinstance(other, Matrix):
            assert self.size == other.size, ShapeException(self.size, other.size)
            for i in range(self.size):
                for j in range(self.size):
                    for k in range(self.size):
                        r[i,j] += self[i,k] * other[k,j]
        if isinstance(other, Vector):
            assert self.size == len(other)
            r = Vector(self.size)
            for i in range(self.size):
                r[i] = sum([self.matrix[i][j] * k for j, k in enumerate(other)])
            return r
        else:
            for i in range(self.size):
                for j in range(self.size):
                    r[i, j] = self.matrix[i][j] * other
        return r

    def __len__(self):
        return self.size
    
    def __str__(self) -> str:
        s = ''
        for row in self.matrix:
            s += '|' + ',\t'.join([str(i) for i in row]) + '|' + '\n'
        return s


def zeros(n: int):
    return Matrix([[0] * n for _ in range(n)])

def eye(n: int):
    m = zeros(n)
    for i in range(n):
        m[i,i] = 1
    return m

def getP(m: Matrix):
    n = len(m)
    p = eye(n)
    m_copy = Matrix(m.matrix)
    for i in range(n):
        if m[i, i] == 0:
            for  j in range(i,n):
                if m[i, j] != 0:
                    p.swap(i,j)
                    m_copy.swap(i,j)
                    break
    return p

def dot(v1: Vector, v2: Vector) -> Matrix:
    assert len(v1) == len(v2)
    n = len(v1)
    l = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            l[i][j] = v1[i] * v2[j]
    return Matrix(l)


if __name__ == '__main__':
    A = Matrix([
        [2,2,3],
        [4,5,6],
        [7,8,9]
    ])
    print(A.extend(3))
    print(A.T)