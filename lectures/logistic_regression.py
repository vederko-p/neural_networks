
# 08.09

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
from stoh_grad_batch import StochasticGradientBatch


class LogisticRegression():
    def fit(self, X, Y, method, Q, h, batch_size, eps=1e-5):
        self.w, self.qx = method.fit(X, Y, Q, h, eps, batch_size)

    def predict(self, X):
        return logistic_sigmoid(X@self.w.T)


def logistic_sigmoid(z):
    # z - numpy array
    return 1 / (1 + np.exp(-z))


def logistic_Q(X, Y, w):
    p = logistic_sigmoid(X@w.T)
    return -1/Y.shape[0] * np.sum(Y*np.log(p) + (1-Y)*np.log(1-p))


if __name__ == '__main__':

    # ---| Данные |---
    data = pd.read_csv(r'data/binary_classification.csv')
    X, Y = data.values[:, :2], data.values[:, -1]

    # ---| Визуализация |---
    fig, ax = plt.subplots()
    for k in np.unique(Y):
        ax.scatter(X[:, 0][Y==k], X[:, 1][Y==k], s=10)
    # plt.show()

    # ---| Выборка |---
    X_train, Y_train = X[:800, :], Y[:800].reshape(-1, 1)
    X_test, Y_test = X[800:, :], Y[800:].reshape(-1, 1)

    # ---| Модель и обучение |---
    model = LogisticRegression()  # модель
    method = StochasticGradientBatch()  # метод
    model.fit(X=X_train, Y=Y_train, method=method, Q=logistic_Q, h=1, batch_size=10, eps=1e-5)

    # ---| Проверка на тестовой выборке |---
    pred = model.predict(X_test)
    colors = ['blue', 'red']
    edge = 0.6
    color_choice = pred > edge

    print(model.w)

    for i in range(X_test.shape[0]):
        color = colors[int(color_choice[i])]
        ax.scatter(X_test[i, 0], X_test[i, 1], color=color, s=50, linewidths=1, edgecolor='black')
    plt.show()

    plt.plot(model.qx)
    plt.show()
