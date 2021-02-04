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
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for when you create your own dendrogram

def kmeans(X, k, max_iter=100):
    muselect = 0
    mu = np.random.rand(k, len(X[0]))
    if muselect == 1:
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        mu = np.random.rand(k, len(X[0]))
        mu += mins
        mu *= maxs
    elif muselect == 0:
        X_shuffle = np.array(X)
        np.random.shuffle(X_shuffle)
        mu = X_shuffle[0:k]
    r = np.empty(len(X))
    r_new = np.ones(len(X))
    iteration = 0
    error=0.
    while not np.array_equal(r,r_new) and iteration < max_iter:
        error=0.
        iteration+=1
        print('Peforming iteration ' + str(iteration) + ':')
        r = r_new
        dis = (-2 * X @ mu.T + (np.sum(X ** 2, axis=1) + np.sum(mu ** 2, axis=1)[:, np.newaxis]).T)**0.5
        r_new = np.argmin(dis, axis=1)
        print('Changed cluster memberships: ' +  str(np.count_nonzero(r_new-r)) + ' datapoints')
        for i in range(0, k):
            if(len(np.where(r_new==i)[0])>0):
                mu[i] = np.sum(X[np.where(r_new==i)], axis=0)/len(np.where(r_new==i)[0])
                error += np.sum(dis[np.where(r_new==i),i])
            else:
                mu[i] = np.random.rand(len(mins))
        print('Loss: ' + str(error))
    return mu, r_new, error
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
    #reconstruct original data
    k_org = max(r)+1
    org_err = 0.
    kmus = np.zeros((k_org,len(X[0])))
    #calculate cluster centers
    for i in np.unique(r):
        if(len(np.where(r==i)[0])>0):
            kmus[i] = np.sum(X[np.where(r==i)], axis=0)/len(np.where(r==i)[0])
            #calculate distances
    org_dis = (-2 * X @ kmus.T + (np.sum(X ** 2, axis=1) + np.sum(kmus ** 2, axis=1)[:, np.newaxis]).T)**0.5
    #calculate errors
    for i in np.unique(r):
        if(len(np.where(r==i)[0])>0):
            org_err += np.sum(org_dis[np.where(r==i),i])
    #create return variable
    R = np.array([r])
    kmloss = np.array([org_err])
    mergeidx = np.array([])
    costs = np.zeros((k_org,k_org))
    #for each combination of clusters (one way, not both ways) calculate cost of merging
    for first in range(0, k_org-1):
        if len(np.where(r==first)[0])>0:
            for second in range(first+1, k_org):
                if len(np.where(r==second)[0])>0:
                    if first != second:
                        numfirst = len(np.where(r==first)[0])
                        numsecond = len(np.where(r==second)[0])
                        #calculating weighted new clustercenter
                        newmu = (kmus[first]*numfirst+kmus[second]*numsecond)/(numfirst+numsecond)
                        newmus = np.append(kmus, [newmu], axis=0)
                        new_r = np.array(r)
                        new_r[np.where(r==first)] = k_org
                        new_r[np.where(r==second)] = k_org
                        dis = (-2 * X @ newmus.T + (np.sum(X ** 2, axis=1) + np.sum(newmus ** 2, axis=1)[:, np.newaxis]).T)**0.5
                        costs[first, second] = np.sum(dis[np.arange(dis.shape[0]),new_r])
    #Choose cheapest merge
    lowest_cost = np.min(costs[np.nonzero(costs)])
    indx = np.asarray(np.where(costs == lowest_cost)).flatten()
    merge_r = np.array(r)
    merge_r[np.where(merge_r==indx[0])] = k_org
    merge_r[np.where(merge_r==indx[1])] = k_org
    #preparing output
    mergeidx = np.array([indx])
    kmloss = np.append(kmloss, lowest_cost)
    #Recursion if needed:
    if len(np.unique(merge_r))>1:
        #R = np.append(R, [merge_r], axis=0)
        R_n, loss_n, mergeidx_n = kmeans_agglo(X, merge_r)
        mergeidx = np.append(mergeidx, mergeidx_n, axis=0)
        kmloss = np.append(kmloss, loss_n[1:])
        R = np.append(R, [merge_r], axis=0)
    return(R, kmloss, mergeidx)

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
    mergeloss = [[i] for i in kmloss[1:]]
    mergeloss -= kmloss[0]
    placeholder_counts = [[1] for i in kmloss[1:]]
    Z = np.append(mergeidx, mergeloss, axis=1)
    Z = np.append(Z, placeholder_counts, axis=1)
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
    dn = dendrogram(Z, ax=axis)
    axis.set_ylabel('Increase in cost', fontsize=20)
    axis.set_xlabel('Cluster number', fontsize=20)
    pass


def norm_pdf(X, mu, C):
    """ Computes probability density function for multivariate gaussian

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: vector for center
    C: covariance matrix

    Output:
    pdf value for each data point
    """

    pass


def em_gmm(X, k, max_iter=100, init_kmeans=False, eps=1e-3):
    """ Implements EM for Gaussian Mixture Models

    Input:
    X: (d x n) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations
    init_kmeans: whether kmeans should be used for initialisation
    eps: when log likelihood difference is smaller than eps, terminate loop

    Output:
    pi: 1 x k matrix of priors
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """

    pass

def plot_gmm_solution(X, mu, sigma):
    """ Plots covariance ellipses for GMM

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """

    pass
