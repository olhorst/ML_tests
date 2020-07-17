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
        # INSERT_CODE

        # Here you have to set the matrices as in the general QP problem
        P = None
        q = None
        G = None
        h = None
        A = None   # hint: this has to be a row vector
        b = None   # hint: this has to be a scalar

        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()

        # b =

    def predict(self, X):
        # INSERT_CODE

        return self


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
    # INSERT CODE
    pass


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
    if Y.isinstance(bool) and Y is False:
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
    def __init__(self, layers=None, scale=.1, p=None, lr=1e-3, lam=None):
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
        Z = mask * (X @ W + b).clamp(0)
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
        Xtrain, ytrain = X[I[:n]], y[I[:n]]
        Xval, yval = X[I[n:]], y[I[n:]]

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