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
    def __init__(self, kernel='linear', kernelparameter=3, regularization=1):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

    def getTrainK(self, X, kernel, kernelparam=3.):
        if kernel == 'linear':
            return X.T*X
        if kernel == 'polynomial':
            return (X.T*X+1)**kernelparam
        if kernel == 'gaussian':
            return np.exp(-1 * ((X.T - X) ** 2) / 2 * kernelparam ** 2)
            pass
        print("Faulty Kernel")
        pass

    def getPredK(self, X, kernel, kernelparam=3.):
        if kernel == 'linear':
            return self.trainX.T*X
        if kernel == 'polynomial':
            return (self.trainX.T*X+1)**kernelparam
        if kernel == 'gaussian':
            return np.exp(-1 * ((self.trainX.T - X) ** 2) / 2 * kernelparam ** 2)
            pass
        print("Faulty Kernel")
        pass

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
        self.alpha = np.linalg.inv(self.getTrainK(X, self.kernel, self.kernelparameter)+self.regularization*np.eye(len(X)))@y
        return self

    def predict(self, X):
        self.m = np.sum(self.alpha*self.getPredK(X, self.kernel, self.kernelparameter),axis=1)
        return self.m
