# Numerical Algorithms for Solving Linear Algebraic Equations
### Foreword
I apologize for the quality of this text and documentation. I am from Russia, and I learned English on my own, and the mathematical language that they give us may differ from the Western one. If you have comments, write them in issues.
### Motivation
As part of my student work, I needed to implement four numerical methods for solving linear systems. Two of them are exact, two are iterative.
QR and LU methods are actually matrix decomposition methods, but the resulting matrices are convenient for solving linear systems and allow you to do this in O(n)
### Implemented Methods
1. LU decomposition - https://en.wikipedia.org/wiki/LU_decomposition
2. QR decomposition - https://en.wikipedia.org/wiki/QR_decomposition
3. Fexid point iteration - https://en.wikipedia.org/wiki/Fixed-point_iteration
4. Seidel method - https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method#:~:text=In%20numerical%20linear%20algebra%2C%20the,a%20system%20of%20linear%20equations.
### Tests
To use test modules you must install `numpy` and `prettytable` packages. You can do this using the following terminal commands:
```console
foo@bar:~$ pip3 install numpy
foo@bar:~$ pip3 install prettytable
```