
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


class LR():
    def predict(self, X):
        return X@self.w.T

    def fit(self, X, Y, method, Q, h, eps=1e-5):
        self.w, self.qx = method.fit(X, Y, Q, h, eps)


class Gradient():
    def fit(self, X, Y, Q, h, eps):
        w0 = np.zeros(X.shape[1]).reshape(1, -1)
        qx = np.array([])
        q = Q(X_train, Y_train, w0)
        qx = np.hstack([qx, q])
        while True:
            w1 = w0 - h*Q_grad(Q, X, Y, w0)
            if (w1 - w0)@(w1 - w0).T <= eps * (1 + w1@w1.T):
                break
            w0 = w1.copy()
            q = Q(X_train, Y_train, w0)
            qx = np.hstack([qx, q])
        return w0, qx


def Q(X, Y, w):
    return (0.5/X.shape[0] * (Y - X@w.T).T @ (Y - X@w.T))[0, 0]


def Q_grad(Q, X, Y, w, eps=1e-5):
    grad = np.array([])
    for i, wi in enumerate(w.flatten()):
        ww = w.copy()
        ww[0, i] = wi + eps
        t = (Q(X, Y, ww) - Q(X, Y, w)) / eps
        grad = np.hstack([grad, t])
    return grad


if __name__ == '__main__':

    # ---| Выборка |---
    n = 1000
    np.random.seed(0)
    X_train = np.arange(n)/n + 0.1*np.random.randn(n)
    Y_train = 10*np.arange(n)/(n+1) + 0.1*np.random.randn(n)
    X_train = X_train.reshape(-1, 1)
    Y_train = Y_train.reshape(-1, 1)

    # ---| Визуализация |---
    fig, gs = plt.figure(figsize=(9, 4)), gridspec.GridSpec(1, 2)
    ax = []
    for i in range(2):
        ax.append(fig.add_subplot(gs[i]))

    ax[0].plot(X_train, Y_train, '.', label='Данные')
    ax[0].set_title('Регрессия')

    # ---| Модель и обучение |---
    X_train = np.hstack([np.ones(n).reshape(-1, 1), X_train])

    linear_regr = LR()  # модель
    method = Gradient()  # метод обучения
    linear_regr.fit(X=X_train, Y=Y_train, method=method, Q=Q, h=0.1, eps=1e-5)  # обучение

    # ---| Предсказание |---
    X_pred = np.linspace(-0.2, 1.5, 100).reshape(-1, 1)
    ones = np.ones(X_pred.shape[0]).reshape(-1, 1)
    X_pred = np.hstack([ones, X_pred])
    Y_pred = linear_regr.predict(X_pred)

    # ---| Визуализация |---
    ax[0].plot(X_pred[:, 1], Y_pred, label='Модель')
    ax[0].legend()

    ax[1].plot(linear_regr.qx)
    ax[1].set_title('Функционал качества')

    plt.show()
