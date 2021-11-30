
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


data = pd.read_csv(r"data/svm_data.csv")
X, Y = data.values[:, :-1], data.values[:, -1]

fig, ax = plt.subplots()
for k in np.unique(Y):
    ax.scatter(X[:, 0][Y==k], X[:, 1][Y==k])
# plt.show()


scalar_x = X@X.T
scalar_y = Y.reshape(-1, 1)@Y.reshape(1, -1)


def func(lambd, sc_x, sc_y):
    # X - мтарица скалярных произведений
    # Y - матрица умножений y
    # lambd - numpy матрица-вектор размерности l; [[lambd1, lambd2, ..., lambdl]]
    l = lambd.T@lambd
    return - lambd.sum() + 0.5*(l*sc_y*sc_x).sum()


def linear_constr(lambd, y):
    # lambd - numpy матрица-вектор
    # y numpy-матрица-столбец
    return (lambd@y).sum()


cons = ({'type': 'eq', 'fun': lambda l: linear_constr(l, Y.reshape(-1, 1))})
bnds = [(0, 1) for i in range(10)]


lmbd = np.zeros_like(Y).reshape(1, -1)
res = minimize(func, lmbd, args=(scalar_x, scalar_y),
               bounds=bnds, constraints=cons)

print(res.x)
