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
    ''' your code here '''


def cv(X, y, method, params, loss_function=zero_one_loss, nfolds=10, nrepetitions=5):
    ''' your header here!
    '''
    return method

  
class krr():
    ''' your header here!
    '''
    def __init__(self, kernel='linear', kernelparameter=1, regularization=1):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

    def getTrainK(self, X, kernel, kernelparam=1.):
        if kernel == 'linear':
            return X.T*X
        if kernel == 'polynomial':
            return (X.T*X+1)**kernelparam
        if kernel == 'gaussian':
            return (np.exp(-1*((X.T-X)**2)/(2 * np.square(kernelparam))))
        if kernel == 'test':
            return (((X.T-X)**2))
        print("dafuq")
        pass
    
    def getPredK(self, X, kernel, kernelparam=1.):
        if kernel == 'linear':
            return self.trainX.T*X
        if kernel == 'polynomial':
            return (self.trainX.T*X+1)**kernelparam
        if kernel == 'gaussian':
            return (np.exp(-1*((self.trainX.T-X)**2)/(2 * np.square(kernelparam))))
        print("dafuq")
        pass
    
    #efficient leave one out cross validation
    def eloocv(self, K, Y):
        eigval, U = la.eig(K)
        U = U.real
        L = np.diag(eigval).real
        UL = np.dot(U, L)

        C3 = np.logspace(-5,5,100)
        I3 = np.eye(len(L))
        CI3 = np.einsum('ij,k->kij', I3, C3)

        CI3L = CI3 + L
        dt = np.dtype(np.float32)
        apinv = list(map(lambda n: la.pinv(n), CI3L))
        apinv = np.asarray(apinv,dtype=dt) 

        ULCI3 = np.einsum('lj,ijk->ilk',UL,apinv)
        ULCI3UT = np.einsum('ikj,jl->ikl', ULCI3, U.T)
        S = ULCI3UT
        Sdiag = np.einsum('kii->ki', S)
        err = np.mean(np.square((Y-np.dot(S, Y))*((1-Sdiag)**-1)), axis=1)
        
        return C3[np.where(err==min(err))]

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
        self.K = self.getTrainK(X, self.kernel, self.kernelparameter)
        if self.regularization==0:
            self.regularization = self.eloocv(self.K, y)
        self.alpha = np.linalg.inv(self.K+self.regularization*np.eye(len(X)))@y
        return self

    def predict(self, X):
        self.m = np.sum(self.alpha*self.getPredK(X, self.kernel, self.kernelparameter),axis=1)
        return self.m
