'''
Vector
======

Implementation of a vector class with the methods necessary to solve a system of linear equations in these ways
'''

from typing import Iterable, Union, overload

class Vector:
    '''
    A class encapsulating an array of constant length and mathematical operations on it
    '''
    def __init__(self, n: int):
        '''
        params
        ------
        n: int
            length of vector
        '''
        self.body = [0.] * n
        self.len = n
    
    @staticmethod
    def fromIterable(i: Iterable[float]):
        '''
        Creating a vector from an iterable object

        params
        ------
        i: Iterable[flaot]
            iterable object
        '''
        body = list(i)
        l = len(body)
        array = Vector(l)
        array.body = body
        return array

    def __add__(self, other: 'Vector'):
        '''
        Piecemeal sum of vetors
        '''
        return Vector.fromIterable([i + j for i, j in zip(self.body, other)])
    
    def __sub__(self, other: 'Vector'):
        '''
        Piecemeal substraction of vetors
        '''
        return Vector.fromIterable([i - j for i, j in zip(self.body, other)])
    

    def __truediv__(self, other: float) -> 'Vector':
        '''
        Non-integer division of all components of a vector by a number
        '''
        return Vector.fromIterable([i / other for i in self.body])

    def __mul__(self, other: Union[float, 'Vector']):
        '''
        Piecemeal multiplication of vetors (not matrix multiplication)
        '''
        if isinstance(other, Vector):
            return Vector.fromIterable([i * j for i, j in zip(self.body, other)])
        return Vector.fromIterable([i * other for i in self.body])

    def dot(self, other: 'Vector') -> float:
        '''
        Scalar multiplication of vectors
        '''
        return sum(self * other) ** 0.5
    
    def abs(self) -> float:
        '''
        Absolute value of vector
        '''
        return sum([i*i for i in self.body]) ** 0.5

    def __iter__(self):
        return self.body.__iter__()

    @overload
    def __getitem__(self, ind: slice) -> 'Vector': ...

    @overload 
    def __getitem__(self, ind: int) -> float: ...

    def __getitem__(self, ind: 'int | slice') -> float:
        if isinstance(ind, slice):
            return Vector.fromIterable(self.body[ind])
        return self.body[ind]
    
    def __setitem__(self, ind: int, val: float):
        self.body[ind] = val

    def copy(self):
        '''
        Copy of vector
        '''
        return Vector.fromIterable(self.body)

    def __str__(self):
        return '('+ ', '.join([str(i) for i in self.body]) + ')'
    
    def __len__(self):
        return len(self.body)


def ort(n: int, e: int):
    v = Vector(n)
    v[e] = 1
    return v