
import numpy as np


X = np.array([[1, 0, 0], [0, 1, 0],
              [0, 0, 1], [1, 1, 0],
              [0, 1, 1], [1, 0, 1],
              [1, 1, 1], [0, 0, 0]])

y = np.array([0, 0, 1, 0, 1, 0, 0, 1])


def sigm(x):
    # x - np.array
    return 1 / (1 + np.exp(-x))


def u(x, w):
    # x - NumPy array
    # w - NumPy array
    return sigm(x@w)


def a(u, w):
    # u - NumPy array
    # w - NumPy array
    return sigm(u@w)


def back_prop(X, y, h, t, n):
    # h - кол-во нейронов скрытого слоя
    # t - шаг градиента
    # n - кол-во итераций
    l, m = X.shape
    w00 = np.ones((m, h))
    w10 = np.ones((h, 1))
    while n:
        i = np.random.randint(l)
        xi, yi = X[i], y[i]
        u0 = u(xi, w00)
        a0 = a(u0, w10)
        # w0 grad step
        dw0 = (a0-yi)*a0*(1-a0)*w10.reshape(-1, h)*u0*(1-u0[0])
        f, g = np.meshgrid(dw0, xi.reshape(-1, 1))
        w00 -= t*f*g
        # w1 grad step
        dw1 = ((a0 - yi)*a0*(1 - a0)*u0).reshape(h, -1)
        w10 -= t*dw1
        n -= 1
    return w00, w10


def predict(X, w0, w1):
    y = np.array([])
    return np.array([a(u(x, w0), w1) for x in X])


if __name__ == '__main__':

    t = back_prop(X, y, 2, 0.1, 7000)
    w0, w1 = t

    test = predict(X, w0, w1)
    print('результаты по тестовой выборке:')
    print(test)

    print('\nвеса:')
    print(w0)
    print(w1)
