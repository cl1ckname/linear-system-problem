'''
Matrix
======
A module implementing a class of a square matrix 
with the necessary methods (classes) and functions 
for solving linear systems
'''

from typing import List, Union, overload
from copy import deepcopy

from .vector import Vector

class ShapeException(Exception):
    '''Exception reporting size mismatch
      
    shape1: int
        size of first object
    shape2: int 
        size of second object
    '''
    def __init__(self, shape1: int, shape2: int):
        self.message = f'shapes incompatible ({shape1}, {shape2})'
    def __str__(self):
        return self.message

class Matrix:
    '''
    A class of a square matrix encapsulating 
    a square array and methods of working with 
    it, as well as some things are necessary for 
    solving linear systems by implemented methods
    '''
    def __init__(self, i: List[List[float]]):
        '''
        params
        ------
        i: List[List[float]]
            square array initializing the matrix
        '''
        assert len(i)
        assert len(i) == len(i[0])
        self.matrix = deepcopy(i)
        self.size = len(i)

    def abs(self, t: int = 0) -> float:
        '''
        Matrix norm in four realizations

        params
        ------
        t: int (from 0 to 3)  

        Type of norm  

        0 - Max norm. 
            (max(a_ij), a_ij <= A)  
        1 - p-1 vector norm 
            (max_i(Sum_0^n(a_ij)))  
        2 - p-inf norm 
            (max_j(Sum_0^n(a_ij)))  
        3 - euclyd norm 
            sum(a_ij) ^ 0.5  
        '''
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

    def swap(self, i: int, j: int):
        '''
        Swaps i'th and j'th matrix rows

        params
        ------
        i: int
            first row
        j: int
            second row
        '''
        self.matrix[i], self.matrix[j] = self.matrix[j], self.matrix[i]

    def extend(self, n: int) -> 'Matrix':
        '''
        Expands the matrix to the desired size, with the original matrix being in the lower left corner.
        Used in QR decomposition.

        params
        ------
        n: int
            new shape 
        '''
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
    def T(self): # yep, numpy style
        '''
        Matrix transposing. Returns new matrix. 
        '''
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
        '''
        Getting an element of a matrix, its row or column
        params combination
        ------------------
        pos: tuple[int, int]
            return matrix element with `pos[0]` and `pos[1]` indexes
        pos: tuple[slice, int]
            return matrix column with `pos[1]` index
        pos: int
            return matrix row with `pos` index
        '''
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
        '''
        Set matrix element value.

        params
        ------
        pos: tuple[int, int]
            inexes of element
        '''
        i,j = pos
        assert i < self.size
        assert j < self.size
        self.matrix[i][j] = v

    def __sub__(self, other: 'Matrix'):
        '''
        Piecemeal subtracts the values of matrix elements
        '''
        assert self.size == other.size, ShapeException(self.size, other.size)
        r = zeros(self.size)
        for i in range(self.size):
            for j in range(self.size):
                r[i,j] = self.matrix[i][j] - other[i, j]
        return r
    
    def __add__(self, other: 'Matrix'):
        '''
        Piecemeal adds the values of matrix elements
        '''
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
        '''
        Matrix product according to the rules of the matrix product

        1. Matrix * Matrx => Matrix
        2. Matrix * number => Matrix
        3. Matrix * Vector => Vector (Only multiplication on the right)
        '''
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
        '''
        The size of a square matrix can be characterized by one number - the number of rows or columns
        '''
        return self.size
    
    def __str__(self) -> str:
        '''
        Returns matrix in pretty to print form
        '''
        s = ''
        for row in self.matrix:
            s += '|' + ',\t'.join([str(i) for i in row]) + '|' + '\n'
        return s


def zeros(n: int):
    '''
    Return zero-matrix of given size

    param
    -----
    n: int
        size of matrix
    '''
    return Matrix([[0] * n for _ in range(n)])

def eye(n: int):
    '''
    Returns a unit matrix - a matrix in which the diagonal elements are equal to one, the rest to zero

    params
    ------
    n: int
        size of matrix
    '''
    m = zeros(n)
    for i in range(n):
        m[i,i] = 1
    return m

def getP(m: Matrix):
    '''
    Get a permutation matrix for a given matrix. 
    Multiplication on the right by this matrix guarantees that 
    there will be no zeros on the diagonal of the 
    multiplication result (if the original matrix does not contain zero columns)

    params
    ------
    m: Matrix
        the matrix for which the permutation matrix will be constructed
    '''
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
    '''
    The product of a "row vector" by a "column vector" according to the rules of matrix multiplication.

    params:
    v1: Vector
        row vector
    v2: Vector
        column vector
    '''
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