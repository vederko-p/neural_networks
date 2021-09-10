
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from stoh_grad_batch import LR, StochasticGradientBatch
from gradient import Q
import random


def true_func(x):
    return np.cos(1.5*np.pi*x)


def make_Folds(X, Y, n):
    # t = make_Folds(X, Y, 2)
    # print(t[1][0][1])  # 1st fold; train; for y
    # =====================
    # X, Y - NumPy matrices
    folds = []  # [[train_1, test_1], ..., [train_n, test_n]]:
    # [
    # [(X_1_tr, Y_1_tr), (X_1_ts, Y_1_ts)],
    # ...,
    # [(X_n_tr, Y_n_tr), (X_n_ts, Y_n_ts)]
    # ]
    fold_len = int(len(Y)/n)
    for i in range(n-1):
        l, r = i*fold_len, (i+1)*fold_len
        test = (X[l:r, :], Y[l:r, :])
        train = np.vstack([X[:l, :], X[r:, :]]), np.vstack([Y[:l, :], Y[r:, :]])
        folds.append([train, test])
    l = (n-1)*fold_len
    test = (X[l:, :], Y[l:, :])
    train = (X[:l, :], Y[:l, :])
    folds.append([train, test])
    return folds


def cross_val(class_model, lmethod, X0, Y0, nFolds, shuffle=False):
    # class_model - class
    if shuffle:
        s = X0.shape[0]
        indx = np.array(random.sample(range(s), s))
        X, Y = X0[indx], Y0[indx]
    else:
        X, Y = X0, Y0
    folds = make_Folds(X, Y, nFolds)
    mse = np.array([])
    for fold in folds:
        model = class_model()
        X_train, Y_train, X_test, Y_test = fold[0][0], fold[0][1], fold[1][0], fold[1][1]
        model.fit(X=X_train, Y=Y_train, method=lmethod, Q=Q, h=0.5, batch_size=15, eps=1e-5)
        predict = model.predict(X_test)
        msei = np.mean((predict - Y_test)**2)
        mse = np.hstack([mse, msei])
    return mse


def make_poly_data(x, n):
    # x - NumPy array
    X = np.ones_like(x).reshape(-1, 1)
    for i in range(1, n+1):
        new_pow = (x**i).reshape(-1, 1)
        X = np.hstack([X, new_pow])
    return X


if __name__ == '__main__':

    # ---| Кросс-валидация |---
    # method = StochasticGradientBatch()
    # cvs = cross_val(LR, method, X, Y, 5, shuffle=True)
    # print(cvs)

    # ---| Визуализация |---
    fig, gs = plt.figure(figsize=(9, 4)), gridspec.GridSpec(2, 2)
    # данные:
    np.random.seed(3)
    x = np.random.uniform(0, 1, 100)
    y = true_func(x) + 0.2 * np.random.uniform(0, 1, 100)
    Y = y.reshape(-1, 1)
    # Отображение исх. данных + модели при некоторых степенях:
    degrees = [5, 8, 20]
    method = StochasticGradientBatch()
    ax = []
    for i in range(3):
        ax.append(fig.add_subplot(gs[i]))
        ax[i].plot(x, y, '.')
        ax[i].set_title('Число степеней свободы: ' + str(degrees[i]))
        model = LR()
        X = make_poly_data(x, degrees[i])
        model.fit(X=X, Y=Y, method=method, Q=Q, h=0.5, batch_size=20, eps=1e-7)
        x_p0 = np.linspace(-0.05, 1.05, 100)
        x_p1 = make_poly_data(x_p0, degrees[i])
        y_p = model.predict(x_p1)
        plt.plot(x_p0, y_p)
    ax.append(fig.add_subplot(gs[3]))
    # зависимость величины ошибки от числа степеней:
    max_degree = 20
    degrees = range(1, max_degree+1)
    cvss = np.array([])
    nFolds = 5
    print('Всего степеней: ' + str(max_degree))
    for degree in degrees:
        X = make_poly_data(x, degree)
        model = LR()
        cvs = np.mean(cross_val(LR, method, X, Y, nFolds, shuffle=True))
        cvss = np.hstack([cvss, cvs])
        print('Обработано: ' + str(degree))
    # график зависимости ошибки от числа степеней свободы:
    ax[-1].set_title('Зависимость MSE от числа степеней полинома')
    ax[-1].plot(degrees, cvss, color='orange', marker='s')
    ax[-1].set_xticks(degrees)
    plt.show()
