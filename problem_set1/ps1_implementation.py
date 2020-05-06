""" sheet1_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Write the functions
- pca
- gammaidx
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la


class PCA:
    def __init__(self, Xtrain):
        self.C = np.cov(Xtrain.T)
        self.D, self.U = la.eig(self.C)

    def project(self, Xtest, m):
        mu = np.average(Xtest, axis=0)
        Z = (Xtest-mu) @ self.U[:, :m]
        return Z

    def denoise(self, Xtest, m):
        Z = self.project(Xtest, m)
        mu = np.average(Xtest, axis=0)
        Y = mu + Z @ self.U[:, :m].T
        return Y


def pca(X, m):
    ''' your header here!
    '''

def gammaidx(X, k):
    n, d = X.shape
    dist = np.full((n, n), -1.0)
    c1 = 0
    y = np.full(n, -1.0)
    kn = np.zeros(k)
    for node in X:
        c2 = 0
        for neigh in X:
            relpos = X[c2] - X[c1]
            dist[c1, c2] = pow(sum(pow(relpos,2)),0.5)
            c2 = c2+1
        y[c1] = np.mean(np.sort(dist[c1])[1:k+1])
        c1 = c1+1
    return y
    


def lle(X, m, n_rule, param, tol=1e-2):
    ''' your header here!
    '''
    print('Step 1: Finding the nearest neighbours by rule ' + n_rule)
    
    print('Step 2: local reconstruction weights')
    
    print('Step 3: compute embedding')
