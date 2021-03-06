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
import numpy.linalg as la
from numpy.random import default_rng
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D  # for when you create your own dendrogram
from scipy import stats

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
    pdf: pdf value for each data point
    logpdf: log pdf value for each data point
    """

    def log_pdf(X, mu, C):
        n, d = X.shape
        inv = la.solve(C, (X - mu).T).T
        maha = np.einsum('ij,ij->i', (X - mu), inv)
        # Directly calculates log(det(C)), bypassing the numerical issues
        # of calculating the determinant of C, which can be very close to zero
        _, logdet = la.slogdet(C)
        log2pi = np.log(2 * np.pi)
        return -0.5 * (d * log2pi + logdet + maha)

    logpdf = log_pdf(X, mu, C)
    return np.exp(logpdf), logpdf


def plot_usps(mu):
    fig, ax = plt.subplots(2, 5)
    fig.set_size_inches(10, 3)
    for i in range(10):
        ax[i // 5, i % 5].imshow(np.reshape(mu[i], [16, 16]))
    plt.show()


def em_gmm(X, k, max_iter=100, init_kmeans=False, tol=1e-4, plot_solution=False):
    """ Implements EM for Gaussian Mixture Models

    Input:
    X: (n x d) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations
    init_kmeans: whether kmeans should be used for initialisation
    tol: regularization paramter
    plot_solution: if True, plots intermidiate solutions

    Output:
    pi: k long vector of priors
    mu: (k x d) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    loglik:: Log likelihood
    """
    init_sample = True
    n, d = X.shape
    pi = np.ones(k)
    pi = pi / np.sum(pi)
    mu = np.random.uniform(np.min(X), np.max(X), (k, d))
    x_std = np.std(X)
    sigma = np.repeat(0.6 * x_std * np.eye(d)[np.newaxis], k, axis=0)

    if init_kmeans:
        # 1. Using k-means
        mu, _, _ = kmeans(X, k)
        sigma += np.repeat(tol * np.eye(d)[np.newaxis], k, axis=0)
    elif init_sample:
        rng = default_rng()
        mu = X[rng.choice(n, size=k, replace=False)]
        sigma += np.repeat(0.5 * np.eye(d)[np.newaxis], k, axis=0)
    else:
        sigma += np.repeat(0.5 * np.eye(d)[np.newaxis], k, axis=0)

    loglik = [0]

    log_r = np.zeros((n, k))
    if plot_solution:
        plot_gmm_solution(X, mu, sigma)
    for i in range(max_iter):

        ###### Step 1 - Expectation
        log_pdf = np.zeros([k, n])

        # Calculating the log responsibilities
        # Note: calculating with logarithms is much safer due to numeric
        for c, m, s, p in zip(range(k), mu, sigma, pi):
            pdf, log_pdf[c] = norm_pdf(X, m, s)
            log_r[:, c] = np.log(p) + log_pdf[c]

        loglik.append(np.log(np.sum(np.exp(log_r))))

        log_sum = logsumexp(log_r, axis=1)[:, None]
        log_r = log_r - log_sum
        r = np.exp(log_r)

        ###### Step 2 - Maximizaton
        n_k = np.sum(r, axis=0)

        # Calculating the component weights
        pi = n_k / n

        # Calculating the components means
        mu = ((r.T @ X).T / n_k).T
        X_mu = X[:, np.newaxis] - mu[np.newaxis]

        # Calculating the component covariances
        for j in range(k):
            r_diag = np.diag(r[:, j])
            sigma_k = (X_mu[:, j].T @ r_diag)
            sigma[j] = (sigma_k @ X_mu[:, j]) / n_k[j]

        sigma += np.repeat(tol * np.eye(d)[np.newaxis], k, axis=0)

        if plot_solution:
            plot_gmm_solution(X, mu, sigma)

        # Terminate if the log likelihood converged to an optimum
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
        ellipse1 = Ellipse(xy=(mu[i, 0], mu[i, 1]),
                           width=lambda_[0] * 4,
                           height=lambda_[1] * 4,
                           angle=np.rad2deg(np.arccos(v[0, 0])),
                           facecolor='none',
                           edgecolor='red')
        ellipse2 = Ellipse(xy=(mu[i, 0], mu[i, 1]),
                           width=lambda_[0] * 8,
                           height=lambda_[1] * 8,
                           angle=np.rad2deg(np.arccos(v[0, 0])),
                           facecolor='none',
                           edgecolor='red')
        ax.add_artist(ellipse1)
        ax.add_artist(ellipse2)
    plt.show()


if __name__ == '__main__':
    X = np.array([[0., 1., 0.5, 3., 3.25, 3.5, 3., 3.25, 3.5],
                  [0., 0., 0.5, 0., 0.5, 0., 2., 2.5, 2.]]).T
    gmm = GMM(n_components=3).fit(X)
    labels = gmm.predict(X)
    mus = gmm.means_
    # plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
    # plt.show()
    em_gmm(X, k=3)
    pass
