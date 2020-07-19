""" ps4_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Complete the classes and functions
- svm_qp
- plot_svm_2d
- neural_network
Write your implementations in the given functions stubs!


(c) Felix Brockherde, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2019
"""
from typing import Iterable, Any, Tuple
import scipy.linalg as la
import matplotlib.pyplot as plt
import sklearn.svm
from cvxopt.solvers import qp
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch
from torch.nn import Module, Parameter, ParameterList
from torch.optim import SGD


class svm_qp():
    """ Support Vector Machines via Quadratic Programming """

    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.alpha_sv = None
        self.b = None
        self.X_sv = None
        self.Y_sv = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        # Compute the Gram matrix
        K = buildKernel(X.T, kernel=self.kernel, kernelparameter=self.kernelparameter)
        # construct P, q, A, b, G, h matrices for CVXOPT
        P = cvxmatrix(np.outer(Y, Y) * K)  # diag instead?
        q = cvxmatrix(np.ones(n_samples) * -1)

        if self.C is None:
            G = cvxmatrix(np.diag(np.ones(n_samples) * -1))
            h = cvxmatrix(np.zeros(n_samples))
        else:
            diag1 = np.diag(np.ones(n_samples) * -1)
            diag2 = np.identity(n_samples)
            G = cvxmatrix(np.vstack((diag1, diag2)))
            zero = np.zeros(n_samples)
            C = np.ones(n_samples) * self.C
            h = cvxmatrix(np.hstack((zero, C)))

        A = cvxmatrix(Y, (1, n_samples))
        b = cvxmatrix(0.0)
        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()
        # Support vectors have non zero lagrange multipliers
        mask = alpha > 1e-5  # some small threshold
        self.X_sv = X[mask]
        self.Y_sv = Y[mask]
        self.a = alpha[mask]
        indices = np.arange(len(alpha))[mask]
        b = .0
        for n in range(len(self.a)):
            by = self.Y_sv[n]
            bypred = np.sum(self.a * self.Y_sv * K[indices[n], mask])
            b = b + (by - bypred)
        self.b = b / len(self.a)

    def plot(self, X, Y):
        X_pos = X[np.where(Y == 1)]
        X_neg = X[np.where(Y == -1)]
        plt.scatter(self.X_sv.T[0], self.X_sv.T[1], marker='+', s=300)
        plt.scatter(X_pos.T[0], X_pos.T[1])
        plt.scatter(X_neg.T[0], X_neg.T[1])

    def predict(self, X):
        K = buildKernel(self.X_sv.T, X.T, kernel=self.kernel, kernelparameter=self.kernelparameter)
        ypred = np.zeros(len(X))
        for n in range(len(X)):
            ypred[n] = self.b + np.sum(self.a * self.Y_sv * K[:, n])
        return ypred


# This is already implemented for your convenience
class svm_sklearn():
    """ SVM via scikit-learn """

    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        if kernel == 'gaussian':
            kernel = 'rbf'
        self.clf = sklearn.svm.SVC(C=C,
                                   kernel=kernel,
                                   gamma=1. / (1. / 2. * kernelparameter ** 2),
                                   degree=kernelparameter,
                                   coef0=kernelparameter)

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.X_sv = X[self.clf.support_, :]
        self.y_sv = y[self.clf.support_]

    def predict(self, X):
        return self.clf.decision_function(X)


def plot_boundary_2d(X, y, model):
    X_pos = X[np.where(y == 1)]
    X_neg = X[np.where(y == -1)]
    maxi = np.max(X, axis=0)
    mini = np.min(X, axis=0)
    xm = np.arange(mini[0], maxi[0], 0.01)
    ym = np.arange(mini[1], maxi[1], 0.01)
    xx, yy = np.meshgrid(xm, ym, indexing='xy')
    xy = np.array([xx, yy])
    xy = xy.T
    coordinates = xy.reshape((-1, 2))
    preds = model.predict(coordinates)
    preds = preds.reshape(len(xm), len(ym)) > 0
    plt.figure(figsize=(15, 10))
    plt.contourf(xm, ym, preds.T)
    if hasattr(model, 'X_sv'):
        plt.scatter(model.X_sv.T[0], model.X_sv.T[1], marker='+', c='r', s=300)
    plt.scatter(X_pos.T[0], X_pos.T[1])
    plt.scatter(X_neg.T[0], X_neg.T[1])


def sqdistmat(X, Y=False):
    if Y is False:
        X2 = sum(X ** 2, 0)[np.newaxis, :]
        D2 = X2 + X2.T - 2 * np.dot(X.T, X)
    else:
        X2 = sum(X ** 2, 0)[:, np.newaxis]
        Y2 = sum(Y ** 2, 0)[np.newaxis, :]
        D2 = X2 + Y2 - 2 * np.dot(X.T, Y)
    return D2


def buildKernel(X, Y=False, kernel='linear', kernelparameter=0):
    d, n = X.shape
    if isinstance(Y, bool) and Y is False:
        Y = X
    if kernel == 'linear':
        K = np.dot(X.T, Y)
    elif kernel == 'polynomial':
        K = np.dot(X.T, Y) + 1
        K = K ** kernelparameter
    elif kernel == 'gaussian':
        K = sqdistmat(X, Y)
        K = np.exp(K / (-2 * kernelparameter ** 2))
    else:
        raise Exception('unspecified kernel')
    return K


class neural_network(Module):
    def __init__(self, layers=None, scale=.1, p=None, lr=.1, lam=None):
        super().__init__()
        if layers is None:
            layers = [2, 100, 2]
        self.weights = ParameterList([Parameter(scale * torch.randn(m, n)) for m, n in zip(layers[:-1], layers[1:])])
        self.biases = ParameterList([Parameter(scale * torch.randn(n)) for n in layers[1:]])

        self.p = p
        self.lr = lr
        self.lam = lam
        self.train = False

    def relu(self, X, W, b):
        rng = np.random.RandomState()
        mask = rng.binomial(size=W.size(1), n=1, p=1 - self.p)
        mask = torch.tensor(mask)
        if self.train:
            Z = mask * (X @ W + b).clamp(0)
        else:
            Z = (X @ W + b).clamp(0)
        return Z

    def softmax(self, X, W, b):
        Z = X @ W + b
        Z_exp = torch.exp(Z)
        partition = torch.sum(torch.exp(Z), dim=1, keepdim=True)
        return Z_exp / partition

    def forward(self, X):

        X = torch.tensor(X, dtype=torch.float)
        for is_last_element, (w, b) in signal_last(zip(self.weights, self.biases)):
            if not is_last_element:
                X = self.relu(X, w, b)
            else:
                X = self.softmax(X, w, b)
        return X

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def loss(self, ypred, ytrue):
        m = ypred.shape[0]
        loss = -torch.sum(ytrue * torch.log(ypred)) / m
        return loss

    def fit(self, X, y, nsteps=1000, bs=100, plot=False):
        X, y = torch.tensor(X), torch.tensor(y)
        optimizer = SGD(self.parameters(), lr=self.lr, weight_decay=self.lam)

        I = torch.randperm(X.shape[0])
        n = int(np.ceil(.1 * X.shape[0]))
        Xtrain, ytrain = X[I[n:]], y[I[n:]]
        Xval, yval = X[I[:n]], y[I[:n]]

        Ltrain, Lval, Aval = [], [], []
        for i in range(nsteps):
            optimizer.zero_grad()
            I = torch.randperm(Xtrain.shape[0])[:bs]
            self.train = True
            output = self.loss(self.forward(Xtrain[I]), ytrain[I])
            self.train = False
            Ltrain += [output.item()]
            output.backward()
            optimizer.step()

            outval = self.forward(Xval)
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

        if plot:
            plt.plot(range(nsteps), Ltrain, label='Training loss')
            plt.plot(range(nsteps), Lval, label='Validation loss')
            plt.plot(range(nsteps), Aval, label='Validation acc')
            plt.legend()
            plt.show()


def signal_last(it: Iterable[Any]) -> Iterable[Tuple[bool, Any]]:
    iterable = iter(it)
    ret_var = next(iterable)
    for val in iterable:
        yield False, ret_var
        ret_var = val
    yield True, ret_var