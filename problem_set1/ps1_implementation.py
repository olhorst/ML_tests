""" sheet1_implementation.py

PUT YOUR NAME HERE:
Benjamin Berta
Oliver Horst


Write the functions
- pca
- gammaidx
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
from scipy.sparse import csgraph
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, Xtrain):
        self.C = np.average(Xtrain, axis=0)
        cov = np.cov(Xtrain.T)

        self.D, self.U = la.eig(cov)
        self.D = self.D.real

        arrinds = self.D.argsort()
        self.D = self.D[arrinds[::-1]]
        self.U = self.U[:, arrinds[::-1]]

    def project(self, Xtest, m):
        Z = (Xtest - self.C) @ self.U[:, :m]
        return Z

    def denoise(self, Xtest, m):
        Z = self.project(Xtest, m)
        Y = self.C + Z @ self.U[:, :m].T
        return Y


def calculate_distance(X):
    """
    Gives back the distance matrix of X
    @param X: data
    @return: distance matrix
    """
    return -2 * X @ X.T + np.sum(X ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]


def knn(X, k):
    """
    K-nearest neighbours algorithm
    @param X: data
    @param k: number of neighbours
    @return: indices and distances of the k nearest neighbours
    """
    # Calculate the euclidean distances
    distances = calculate_distance(X)

    # Avoid negative numbers due to numeric error
    distances[distances < 0] = 0

    distances = np.sqrt(distances)
    indices = np.argsort(distances, 0)
    distances = np.sort(distances, 0)

    return indices[1:k + 1, :], distances[1:k + 1, :]


def eps_ball(X, eps):
    """
    Returns neighbours that are inside an aps-radius sphere of the point
    @param X: data
    @param eps: radius of the sphere that is used for the neighbourhood search
    @return: masked indices and distances of the resulting neighbours
    """
    # Calculate the euclidean distances
    distances = calculate_distance(X)

    # Avoid negative numbers due to numeric error
    distances[distances < 0] = 0

    distances = np.sqrt(distances)

    # Create mask
    mask = np.where(distances < eps, 0, 1)

    indices = np.argsort(distances, 0)
    mask = np.take_along_axis(mask, indices, 0)
    distances = np.sort(distances, 0)

    indices = np.ma.masked_array(indices, mask=mask)
    distances = np.ma.masked_array(distances, mask=mask)

    return indices[1:], distances[1:]


def gammaidx(X, k):
    """
    Returns the gamma value for the points in their k-neighbourhood
    @param X: data
    @param k: number of neighbours used in the knn search
    @return: gamma index
    """
    _, distances = knn(X, k)

    y = np.average(distances, axis=0)

    return y


def auc(y_true, y_pred, plot=False):
    """
    Area Under Curve, also called "c-statistic" ("concordance statistic")

    @param y_true: true labels, {-1,1}^n
    @param y_pred: predicted value, [-1,1]
    @param plot: boolean, when true plot the ROC curve
    @return: Area Under Curve
    """
    # Convert data to {0, 1} format
    y_true = np.where(y_true == 1.0, 1, 0)

    # sort values in descending order
    indices_desc = np.argsort(y_pred)[::-1]
    y_true = y_true[indices_desc]
    y_pred = y_pred[indices_desc]

    # Calculate True Positives and False Negatives
    tps = np.cumsum(y_true)
    fps = 1 + np.arange(tps.size) - tps

    # Making sure that the first value is (0,0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    # Calculating False Positive Rate and True Positive Rate
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    # AUC is the area under the ROC curve
    # Using trapezoidal integration
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


def lle(X, m, n_rule, k=None, tol=1e-3, epsilon=None):
    """

    @param X: data points (nxd)
    @param m: dimension of the embedding
    @param n_rule: method used for the neighbour graph creation. options: 'knn', 'eps-ball'
    @param k: number of neighbors
    @param tol: regularization parameter for the local covariance matrices
    @param epsilon: radius of the sphere that is used for the neighbourhood search in eps-ball
    @return: embedding in m-dimensions
    """
    print('Step 1: Finding the nearest neighbours by rule ' + n_rule)
    n, d = X.shape

    if n_rule == "knn":
        if k is None:
            raise ValueError('k must be set if rule "knn" is chosen')
        elif k > len(X):
            raise ValueError('k may not exceed the total amount of datapoints, which is ' + len(X))
        indices, distances = knn(X, k)

    elif n_rule == 'eps-ball':
        if epsilon is None:
            raise ValueError('epsilon must be set if rule "eps-ball" is chosen')
        indices, distances = eps_ball(X, epsilon)
    else:
        raise ValueError('Only knn and eps-ball are excepted as n_rule')

    adj = np.zeros([n, n])
    for i in range(n):
        adj[i, indices[:, i]] = 1

    connected_components, component_indices = csgraph.connected_components(adj)

    if not connected_components == 1:
        raise ValueError('The resulted graph is not connected')

    print('Step 2: local reconstruction weights')

    # Initialize matrix of reconstruction weights
    W = np.zeros([n, n])

    # regularlizer only in case constrained fits are ill conditioned
    if n_rule == "knn" and k <= d:
        tol = 0

    for i in range(n):
        if n_rule == 'eps-ball':
            ind = indices[:, i].compressed()
            k = indices[:, i].count()
        else:
            ind = indices[:, i]

        # shift ith pt to origin
        z = X[ind] - np.tile(X[i], (k, 1))

        # local covariance
        C = z @ z.T

        # regularlization (K>D)
        C = C + np.eye(k) * tol * np.trace(C)

        # solve Cw=1
        w = np.squeeze(la.solve(C, np.ones((k, 1))))

        # enforce sum(w)=1
        w = w / np.sum(w)
        W[i, ind] = np.squeeze(w)

    print('Step 3: compute embedding')

    M = np.eye(n) - W
    M = M.T @ M

    D, V = la.eigh(M)
    Y = V[:, 1:m + 1]

    return Y
