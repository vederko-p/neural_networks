
# 08.09

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from gradient import Q, Q_grad
import random


class LR():
    def predict(self, X):
        return X@self.w.T

    def fit(self, X, Y, method, Q, h, batch_size, eps=1e-5):
        self.w, self.qx = method.fit(X, Y, Q, h, eps, batch_size)


class StochasticGradientBatch():
    def fit(self, X, Y, Q, h, eps, batch_size):
        w0 = np.zeros(X.shape[1]).reshape(1, -1)
        qx = np.array([])
        q = Q(X, Y, w0)
        qx = np.hstack([qx, q])
        l_range = range(X.shape[0])
        while True:
            batch_indxs = random.sample(l_range, batch_size)
            X_batch, Y_batch = X[batch_indxs], Y[batch_indxs]
            w1 = w0 - h*Q_grad(Q, X_batch, Y_batch, w0)
            if (w1 - w0)@(w1 - w0).T <= eps * (1 + w1@w1.T):
                break
            w0 = w1.copy()
            q = Q(X, Y, w0)
            qx = np.hstack([qx, q])
        return w0, qx


if __name__ == '__main__':

    # ---| Выборка |---
    n = 1000
    np.random.seed(0)
    X_train = np.arange(n)/n + 0.1*np.random.randn(n)
    Y_train = 10*np.arange(n)/(n+1) + 0.1*np.random.randn(n)
    X_train = X_train.reshape(-1, 1)
    Y_train = Y_train.reshape(-1, 1)
    X_train = np.hstack([np.ones(n).reshape(-1, 1), X_train])

    # ---| Визуализация |---
    batch_sizes = [1, 10, 30, 50, 100, 200, 300, 500, 700, 1000]
    method = StochasticGradientBatch()  # метод обучения

    fig, gs = plt.figure(figsize=(9, 4)), gridspec.GridSpec(2, int(len(batch_sizes)/2))
    ax = []
    for i in range(len(batch_sizes)):
        ax.append(fig.add_subplot(gs[i]))
        model = LR()  # модель
        model.fit(X=X_train, Y=Y_train, method=method, Q=Q, h=0.1, batch_size=batch_sizes[i], eps=1e-5)  # обучение
        ax[i].plot(model.qx)
        ax[i].set_title("Batch size: " + str(batch_sizes[i]))
    plt.show()
