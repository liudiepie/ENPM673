#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import random

x1, y1, xp1, yp1, x2, y2, xp2, yp2, x3, y3, xp3, yp3, x4, y4, xp4, yp4 = 5, 5, 100, 100, 150, 5, 200, 80, 150, 150, 220, 80, 5, 150, 100, 200

#input matrix
A = np.array([[-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1],
              [0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1],
              [-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*xp2, xp2],
              [0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],
              [-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*xp3, xp3],
              [0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],
              [-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*xp4, xp4],
              [0, 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]])

#compute svd
def compute_svd(A):
    #find eigen values and vectors
    AT = A.T
    ATA = AT.dot(A)
    eigen_values, eigen_vectors = np.linalg.eig(ATA)
    sort_eig = eigen_values.argsort()[::-1]
    new_eigen_values = eigen_values[sort_eig]
    new_eigen_vectors = eigen_vectors[:,sort_eig]
    eigen_vectorsT = new_eigen_vectors.T

    #find U transpose
    AAT = A.dot(AT)
    eigen_values_U, eigen_vectors_U = np.linalg.eig(AAT)
    sort_eig1 = eigen_values_U.argsort()[::-1]
    new_eigen_values_U = eigen_values_U[sort_eig1]
    new_eigen_vectors_U = eigen_vectors_U[:,sort_eig1]
    diag = np.diag((np.sqrt(new_eigen_values_U)))

    #find the sigma matrix
    sigma = np.zeros_like(A).astype(np.float64)
    sigma[:diag.shape[0],:diag.shape[1]]=diag

    #find homography matrix
    H = new_eigen_vectors[:,8]
    H = np.reshape(H,(3,3))
    return eigen_vectorsT, new_eigen_vectors_U, sigma, H

VT, U, S, H = compute_svd(A)

print('U', U, '\n', 'S', S, '\n', 'VT', VT)
#print('SVD: U, S, VT', np.linalg.svd(A))
print('homography', H)

