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
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, Xtrain):
        self.C = np.average(Xtrain, axis=0)
        self.D, self.U = la.eig(self.C)

    def project(self, Xtest, m):
        Z = (Xtest - self.C) @ self.U[:, :m]
        return Z

    def denoise(self, Xtest, m):
        Z = self.project(Xtest, m)
        Y = self.C + Z @ self.U[:, :m].T
        return Y


def pca(X, m):
    ''' your header here!
    '''


def knn(X, k):
    """
    source: https://nycdatascience.com/blog/student-works/machine-learning/knn-classifier-from-scratch-numpy-only/
    @param X:
    @param k:
    @return:
    """
    # Calculate the euclidean distances
    distances = -2 * X @ X.T + np.sum(X ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]

    # Avoid negative numbers due to numeric error
    distances[distances < 0] = 0

    distances = np.sqrt(distances)
    indices = np.argsort(distances, 0)
    distances = np.sort(distances, 0)

    return indices[1:k + 1, :], distances[1:k + 1, :]


def gammaidx(X, k):
    """

    @param X:
    @param k:
    @return:
    """
    indices, distances = knn(X, k)

    y = np.average(distances, axis=0)

    return y


def auc(y_true, y_pred, plot=False):
    """
    Area Under Curve, also called "c-statistic" ("concordance statistic")

    True Positive Rate (TPR)
    False Positive Rate (FPR)
    
    @param y_true: true labels, {-1,1}^n
    @param y_pred: predicted value, [-1,1]
    @param plot: boolean, when true plot the ROC curve
    @return:
    """
    y_true = np.where(y_true == 1.0, 1, 0)

    indices_desc = np.argsort(y_pred)[::-1]
    y_true = y_true[indices_desc]
    y_pred = y_pred[indices_desc]

    # Calculate True Positives and False Negatives
    tps = np.cumsum(y_true)
    fps = 1 + np.arange(tps.size) - tps

    # Making sure that the firs value is (0,0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    # Calculating False Positive Rate and True Positive Rate
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    # AUC is the area under the ROC curve
    c = np.trapz(tpr, fpr)

    if plot:
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    return c


def lle(X, m, n_rule, param, tol=1e-2):
    ''' your header here!
    '''
    print('Step 1: Finding the nearest neighbours by rule ' + n_rule)
    
    print('Step 2: local reconstruction weights')
    
    print('Step 3: compute embedding')
