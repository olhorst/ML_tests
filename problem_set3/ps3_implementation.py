""" ps3_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Write the functions
- cv
- zero_one_loss
- krr
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
import itertools as it
import time
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D


def zero_one_loss(y_true, y_pred):
    ''' your header here!
    '''


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(dicts):
    return (dict(zip(dicts, x)) for x in it.product(*dicts.values()))


def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5):
    ''' your header here!
    '''
    x=y-1
    n, d = X.shape

    best_method = None
    best_err = np.float('inf')
    err_arr = []
    configs = product_dict(params)
    configs, configs_ = it.tee(configs)
    n_configs = sum(1 for x in configs_)
    avg_time = 0.0
    c = 0
    for conf in configs:
        t_start = time.time()
        for i in range(nrepetitions):
            ind = np.arange(0, n)
            np.random.shuffle(ind)

            for j in range(nfolds):

                div, mod = divmod(n, nfolds)
                val_ind = [x for x in range(j * div + min(j, mod), (j + 1) * div + min(j + 1, mod))]
                val_ind = ind[val_ind]

                X_ = np.delete(X, val_ind)
                y_ = np.delete(y, val_ind)
                X_val = X[val_ind]
                y_val = y[val_ind]

                # print('Curent configuration: ' + str(conf))
                meth = method(**conf)
                meth.fit(X_, y_, **conf)
                y_pred = meth.predict(X_val)

                err = loss_function(y_val, y_pred)
                if n_configs != 1 and err < best_err:
                    meth.cv_loss = err
                    best_method = meth
                    best_err = err
                else:
                    err_arr.append(err)

        t_end = time.time()
        c += 1
        if c > 1:
            avg_time = (c-1)/c*avg_time + (t_end-t_start)/c
        else:
            avg_time = t_end-t_start
        print('time remaining: ' + str(int((n_configs - c)*avg_time)) + 's')

        if n_configs != 1:
            best_method.fit(X, y)
        else:
            best_method = method(**conf)
            best_method.fit(X, y, **conf)
            best_method.avg_loss = np.average(err_arr)

    return best_method


class krr():
    ''' your header here!
    '''

    def __init__(self, kernel='linear', kernelparameter=3, regularization=1):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization
        self.trainX = None
        self.alpha = None

    def getTrainK(self, X):
        if self.kernel == 'linear':
            return X.T * X
        elif self.kernel == 'polynomial':
            return (X.T * X + 1) ** self.kernelparameter
        elif self.kernel == 'gaussian':
            X = X.reshape(-1, 1)
            ret = np.exp(-1 * (X.T - X) ** 2/ 2 * self.kernelparameter ** 2)
            return ret
        else:
            print("Faulty Kernel")

    def getPredK(self, X):
        if self.kernel == 'linear':
            return self.trainX.T * X
        elif self.kernel == 'polynomial':
            return (self.trainX.T * X + 1) ** self.kernelparameter
        elif self.kernel == 'gaussian':
            return np.exp(-1 * (self.trainX.T - X) ** 2 / 2 * self.kernelparameter ** 2)
        else:
            print("Faulty Kernel")

    def eloocv(self, K, Y):
        eigval, U = la.eig(K)
        U = U.real
        L = np.diag(eigval).real
        UL = np.dot(U, L)

        C3 = np.logspace(-5, 5, 100)
        I3 = np.eye(len(L))
        CI3 = np.einsum('ij,k->kij', I3, C3)

        CI3L = CI3 + L
        dt = np.dtype(np.float32)
        apinv = list(map(lambda n: la.pinv(n), CI3L))
        apinv = np.asarray(apinv, dtype=dt)

        ULCI3 = np.einsum('lj,ijk->ilk', UL, apinv)
        ULCI3UT = np.einsum('ikj,jl->ikl', ULCI3, U.T)
        S = ULCI3UT
        Sdiag = np.einsum('kii->ki', S)
        err = np.mean(np.square((Y - np.dot(S, Y)) * ((1 - Sdiag) ** -1)), axis=1)

        return C3[np.where(err == min(err))]

    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
        ''' your header here!
        '''
        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization
        self.trainX = X

        K = self.getTrainK(X)
        if self.regularization == 0:
            self.regularization = self.eloocv(K, y)
        mat = K + self.regularization * np.eye(len(X))
        self.alpha = np.linalg.inv(mat) @ y

    def predict(self, X):
        return np.sum(self.alpha * self.getPredK(X), axis=1)
