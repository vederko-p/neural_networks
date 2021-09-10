
# 08.09

import numpy as np
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
