'''
Script that generats two tables
===============================
First table columns
-------------------
n - number of test
x_hat - answer founded by numpy
eps - required error
sim_x - Fixed point result
sim_delta - Absolute error of Fixed point solution
sim_k - number of iterations
(another columns determines same)

Second table columns
--------------------
n - Size of system matrix
eps - Required error
e - amount of conditioned
(another columns determines same)
'''

from prettytable import PrettyTable
from methods.LU import SolveLU
from methods.QR import QRSolve
from methods.Seildel import SeidelSolve
from methods.FixedPointIteration import IterationSolve
from testMatrix import generateTest5, tests

columns1 = ['n', 'x_hat', 'eps', 'sim_x', 'sim_delta', 'sim_k', 'seidel_x', 'seidel_delta', 'seidel_k', 'LU_x', 'LU_delta', 'QR_x', 'QR_delta']
table1 = PrettyTable(columns1)
for n, test in enumerate(tests):
    x = test.x
    lu_x = SolveLU(test.A, test.b)
    qr_x = QRSolve(test.A, test.b)
    lu_delta = (x - lu_x).abs()
    qr_delta = (x - qr_x).abs()
    for p in range(3,7):
        eps = 10**(-p)
        sx, sk = IterationSolve(test.A, test.b, eps)
        sd = (x - sx).abs()
        zx, zk = SeidelSolve(test.A, test.b, eps)
        zd = (x - zx).abs()
        table1.add_row([n, test.x, eps, sx, sd, sk, zx, zd, zk, lu_x, lu_delta, qr_x, qr_delta])
print(table1)
print()

columns2 = ['n', 'eps', 'x_hat', 'e', 'sim_x', 'sim_delta', 'sim_k', 'seidel_x', 'seidel_delta', 'seidel_k', 'LU_x', 'LU_delta', 'QR_x', 'QR_delta']

table2 = PrettyTable(columns2)
for n in range(4, 10):
    for eps in (1e-3, 1e-6):
        tst = generateTest5(n, eps)
        x_hat = tst.x
        LU_x = SolveLU(tst.A, tst.b)
        LU_delta = (x_hat - LU_x).abs()
        QR_x = QRSolve(tst.A, tst.b)
        QR_delta = (x_hat - QR_x).abs()
        for e in (1e-3, 1e-4, 1e-5, 1e-6):
            sim_x, sim_k = IterationSolve(tst.A, tst.b, e)
            sim_delta = (x_hat - sim_x).abs()
            zeydel_x, zeydel_k = SeidelSolve(tst.A, tst.b, e)
            zeydel_delta = (x_hat - zeydel_x).abs()
        table2.add_row([n, eps, x_hat, e, sim_x, sim_delta, sim_k, zeydel_x, zeydel_delta, zeydel_k, LU_x, LU_delta, QR_x, QR_delta])
print(table2)

