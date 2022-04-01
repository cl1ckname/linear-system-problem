from typing import Iterable, Union, overload

class Vector:
    def __init__(self, n: int):
        self.body = [0.] * n
        self.len = n
    
    @staticmethod
    def fromIterable(i: Iterable[float]):
        body = list(i)
        l = len(body)
        array = Vector(l)
        array.body = body
        return array

    def __add__(self, other: 'Vector'):
        return Vector.fromIterable([i + j for i, j in zip(self.body, other)])
    
    def __sub__(self, other: 'Vector'):
        return Vector.fromIterable([i - j for i, j in zip(self.body, other)])
    

    def __truediv__(self, other: float) -> 'Vector':
        return Vector.fromIterable([i / other for i in self.body])

    def __mul__(self, other: Union[float, 'Vector']):
        if isinstance(other, Vector):
            return Vector.fromIterable([i * j for i, j in zip(self.body, other)])
        return Vector.fromIterable([i * other for i in self.body])

    def dot(self, other: 'Vector') -> float:
        return sum(self * other) ** 0.5
    
    def abs(self) -> float:
        return sum([i*i for i in self.body]) ** 0.5

    def __iter__(self):
        return self.body.__iter__()

    @overload
    def __getitem__(self, ind: slice) -> 'Vector': ...

    def __getitem__(self, ind: 'int | slice') -> float:
        if isinstance(ind, slice):
            return Vector.fromIterable(self.body[ind])
        return self.body[ind]
    
    def __setitem__(self, ind: int, val: float):
        self.body[ind] = val

    def copy(self):
        return Vector.fromIterable(self.body)

    def __str__(self):
        return '('+ ', '.join([str(i) for i in self.body]) + ')'
    
    def __len__(self):
        return len(self.body)


def ort(n: int, e: int):
    v = Vector(n)
    v[e] = 1
    return v