""" sheet1_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Write the functions
- usps
- outliers
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import pylab as pl
import random
import importlib
import scipy.io
import matplotlib.pyplot as plt

import ps1_implementation as imp

importlib.reload(imp)


def usps():
    ''' performs the usps analysis for assignment 4'''

    # ----------------------------------- 1 ------------------------------------ #
    usps = scipy.io.loadmat('data/usps.mat')
    data = usps['data_patterns'].T
    n, d = data.shape
    pca = imp.PCA(data)

    # ----------------------------------- 2 ------------------------------------ #
    # Visualize (a) all principal values, (b) the largest 25 principal values both as a bar plots
    fig, ax = plt.subplots(2)
    ax[0].plot(pca.D)
    ax[1].bar(np.arange(25), pca.D[:25])
    plt.show()

    # Visualize the first 5 principal directions as images
    fig_2, ax_2 = plt.subplots(1, 5)
    for i in range(5):
        ax_2[i].imshow(np.reshape(pca.U[:, i], [16, 16]))

    plt.show()

    # ----------------------------------- 3 ------------------------------------ #
    # Adding noise
    gaussian_noise = np.random.randn(n, d)
    data_with_low_noise = data + 0.2 * gaussian_noise
    data_with_high_noise = data + 0.4 * gaussian_noise
    data_with_very_high_noise = np.copy(data)
    data_with_very_high_noise[:5] += gaussian_noise[:5]

    # # plot images with added noise
    # for i in range(10):
    #     fig_3, ax_3 = plt.subplots(1, 4)
    #     ax_3[0].imshow(np.reshape(data[i], [16, 16]))
    #     ax_3[1].imshow(np.reshape(data_with_low_noise[i], [16, 16]))
    #     ax_3[2].imshow(np.reshape(data_with_high_noise[i], [16, 16]))
    #     ax_3[3].imshow(np.reshape(data_with_very_high_noise[i], [16, 16]))
    #     plt.show()
    #     print(i)

    # Calculate the PCA of this data and redo the plots of principal values. Explain the differences to the original
    # spectrum.
    pca_1 = imp.PCA(data_with_low_noise)
    pca_2 = imp.PCA(data_with_high_noise)
    pca_3 = imp.PCA(data_with_very_high_noise)

    fig_4, ax_4 = plt.subplots(4)
    ax_4[0].plot(pca.D[:25])
    ax_4[1].plot(pca_1.D[:25])
    ax_4[2].plot(pca_2.D[:25])
    ax_4[3].plot(pca_3.D[:25])
    plt.show()

    # Denoise the images by reconstruction from projections on the m largest principal components
    m = 10
    Y = pca.denoise(data, m=m)
    Y_1 = pca_1.denoise(data_with_low_noise, m=m)
    Y_2 = pca_2.denoise(data_with_high_noise, m=m)
    Y_3 = pca_3.denoise(data_with_very_high_noise, m=m)

    for i in range(10):
        fig_5, ax_5 = plt.subplots(2, 4)
        ax_5[0, 0].imshow(np.reshape(data[i], [16, 16]))
        ax_5[0, 1].imshow(np.reshape(data_with_low_noise[i], [16, 16]))
        ax_5[0, 2].imshow(np.reshape(data_with_high_noise[i], [16, 16]))
        ax_5[0, 3].imshow(np.reshape(data_with_very_high_noise[i], [16, 16]))

        ax_5[1, 0].imshow(np.reshape(Y[i], [16, 16]))
        ax_5[1, 1].imshow(np.reshape(Y_1[i], [16, 16]))
        ax_5[1, 2].imshow(np.reshape(Y_2[i], [16, 16]))
        ax_5[1, 3].imshow(np.reshape(Y_3[i], [16, 16]))
        plt.show()
        print(i)


def outliers_calc():
    ''' outlier analysis for assignment 5'''

    # np.savez_compressed('outliers', var1=var1, var2=var2, ...)


def outliers_disp():
    ''' display the boxplots'''
    results = np.load('outliers.npz')


def lle_visualize(dataset='flatroll'):
    ''' visualization of LLE for assignment 6'''


def lle_noise():
    ''' LLE under noise for assignment 7'''


if __name__ == '__main__':
    usps()
