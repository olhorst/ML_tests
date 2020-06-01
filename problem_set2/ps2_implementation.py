""" ps2_implementation.py

PUT YOUR NAME HERE:
Benjamin Berta
Oliver Horst


Write the functions
- kmeans
- kmeans_agglo
- agglo_dendro
- norm_pdf
- em_gmm
- plot_gmm_solution

(c) Felix Brockherde, TU Berlin, 2013
    Translated to Python from Paul Buenau's Matlab scripts
"""

from __future__ import division  # always use float division
import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D  # for when you create your own dendrogram

def kmeans(X, k, max_iter=100):
    """ Performs k-means clustering

    Input:
    X: (d x n) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations

    Output:
    mu: (d x k) matrix with each cluster center in one column
    r: assignment vector
    """

    pass


def kmeans_agglo(X, r):
    """ Performs agglomerative clustering with k-means criterion

    Input:
    X: (d x n) data matrix with each datapoint in one column
    r: assignment vector

    Output:
    R: (k-1) x n matrix that contains cluster memberships before each step
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """

    def kmeans_crit(X, r):
        """ Computes k-means criterion

        Input: 
        X: (d x n) data matrix with each datapoint in one column
        r: assignment vector

        Output:
        value: scalar for sum of euclidean distances to cluster centers
        """

        pass
    
    pass



def agglo_dendro(kmloss, mergeidx):
    """ Plots dendrogram for agglomerative clustering

    Input:
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """

    pass


def norm_pdf(X, mu, C):
    """ Computes probability density function for multivariate gaussian

    Input:
    X: (n x d) data matrix with each datapoint in one column
    mu: vector for center
    C: covariance matrix

    Output:
    pdf value for each data point
    """
    n, d = X.shape
    inv = np.linalg.solve(C, (X - mu).T).T
    prod = np.einsum('ij,ij->i', inv, inv)

    y = 1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(C))) * np.exp(-(prod / 2))
    return y


def em_gmm(X, k, max_iter=100, init_kmeans=False, eps=1e-3):
    """ Implements EM for Gaussian Mixture Models

    Input:
    X: (n x d) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations
    init_kmeans: whether kmeans should be used for initialisation
    eps: when log likelihood difference is smaller than eps, terminate loop

    Output:
    pi: k long vector of priors
    mu: (k x d) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """
    init_sample = False
    n, d = X.shape
    pi = np.ones(k)
    pi = pi / np.sum(pi)
    sigma = np.repeat(np.eye(d)[np.newaxis], k, axis=0)
    np.random.seed(0)
    mu = np.random.uniform([0, 0], [4, 2], (k, d))

    if init_kmeans:
        # 1. Using k-means
        mu, _ = kmeans(X, k)
    elif init_sample:
        rng = default_rng()
        mu = X[rng.choice(n, size=k, replace=False)]

    loglik = [0]
    r = np.zeros((n, k))
    for i in range(max_iter):

        ###### Step 1 - Expectation
        for c, m, s, p in zip(range(k), mu, sigma, pi):
            r[:, c] = p * norm_pdf(X, m, s)

        loglik.append(-np.sum(r))
        r = r / np.sum(r, axis=1)[:, None]

        ###### Step 2 - Maximizaton
        n_k = np.sum(r, axis=0)
        pi = n_k / n
        mu = (1 / n_k * np.sum(r[:, np.newaxis] * X[:, :, None], axis=0)).T
        X_mu = X[:, np.newaxis] - mu[np.newaxis]

        outer_prod = r[:, :, np.newaxis, np.newaxis] * np.matmul(X_mu[:, :, :, np.newaxis], X_mu[:, :, np.newaxis])
        sigma = 1 / n_k[:, None, None] * np.sum(outer_prod.swapaxes(0, 1), axis=1)

        tol = np.repeat(1e-6 * np.eye(d)[np.newaxis], k, axis=0)
        sigma += tol
        if np.isclose(loglik[i], loglik[i - 1]):
            break

    return pi, mu, sigma, loglik[-1]


def plot_gmm_solution(X, mu, sigma):
    """ Plots covariance ellipses for GMM

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """
    k = len(mu)

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(mu[:, 0], mu[:, 1], marker='x', c='red')

    for i in range(k):
        lambda_, v = np.linalg.eig(sigma[i])
        ellipse = Ellipse(xy=(mu[i, 0], mu[i, 1]),
                          width=lambda_[0] * 2,
                          height=lambda_[1] * 2,
                          angle=np.rad2deg(np.arccos(v[0, 0])),
                          facecolor='none',
                          edgecolor='red')
        ax.add_artist(ellipse)
    plt.show()

    pass
