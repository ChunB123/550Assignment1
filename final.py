# Problem3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import sklearn.model_selection
import sklearn.linear_model
import sklearn.datasets
from tqdm import tqdm

dataset = sk.datasets.load_digits()
X = dataset.data / 16
y = dataset.target
X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X, y)

reg = sklearn.linear_model.LogisticRegression().fit(X_train, y_train)
y_pred = reg.predict(X_val)
print("Problem3's Acc: ", np.average(y_pred == y_val))

#######################################################
# Problem4
dataset = sk.datasets.load_digits()
X = dataset.data / 16
y = dataset.target
y = (y == 8).astype("int64")
X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X, y)


def Xhat(X):
    return np.insert(X, 0, 1, axis=1)


def sigmoid(u):
    if u > 0:
        return 1 / 1 + np.exp(-u)
    else:
        return np.exp(u) / (1 + np.exp(u))


def binary_cross_entropy(p, q, eps=1e-10):
    q = np.clip(q, eps, 1 - eps)
    return - p * np.log(q) - (1 - p) * np.log(1 - q)


def grad_L(X, y, beta):
    # input is Xhat
    return np.average(np.array([(sigmoid(xi @ beta) - y[idx]) * xi for idx, xi in enumerate(X)]), axis=0)


def eval_L(X, y, beta, bias=1):
    biasTerm = (bias / 2) * (beta @ beta)
    return biasTerm+ np.average([binary_cross_entropy(y[idx], sigmoid(xi @ beta)) for idx, xi in enumerate(X)], axis=0)


def train_gd_model(X, y, alpha=0.01, iter=300):
    X = Xhat(X)
    beta = np.zeros(X.shape[1])
    L_vals = []
    for _ in tqdm(range(iter)):
        beta = beta - alpha * grad_L(X, y, beta)
        L_vals.append(eval_L(X, y, beta))
    return beta, L_vals


beta_G, L_vals_G = train_gd_model(X_train, y_train)
plt.plot(L_vals_G)
plt.show()

result = Xhat(X_val) @ beta_G
y_pred_4 = np.array([1 if sigmoid(r)>0.5 else 0 for r in result])

print("Problem4's acc: ", np.average(y_pred_4 == y_val))
