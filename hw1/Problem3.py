#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#input data
dataset = pd.read_csv('Problem3_data.csv')
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, -1].values

#covariance matrix
mat = np.array((x, y))
age_mean = np.mean(x)
cost_mean = np.mean(y)
means = np.array([[age_mean], [cost_mean]])
N = mat.shape[1]
n_dim = mat.shape[0]
print('n_dim', n_dim)
cov = np.zeros((n_dim, n_dim))
for i in range(n_dim):
    for j in range(n_dim):
        var = 0
        for k in range(N):
            var+=(mat[i][k] - means[i]) * (mat[j][k] - means[j])
        var /= N
        cov[i, j] = var
print('cov', cov)
#print('np cov', np.cov(mat)) #check the accuracy

#eigen_values and eigen_vectors
eigen_values, eigen_vectors = np.linalg.eig(cov)
origin = means
print('eig_v', eigen_vectors)
eig_vec1 = eigen_vectors[0,:]
eig_vec2 = eigen_vectors[1,:]
#standard least squares method
def ls(x, y):
    A = np.stack((x, np.ones((len(x)), dtype = int)), axis=1) #A matrix y = ax + b
    ls_pred = ls_fit(A, y)
    ls_model = A.dot(ls_pred)
    return ls_model

def ls_fit(A, y):
    A_trans = A.transpose()
    ATA = A_trans.dot(A)
    ATY = A_trans.dot(y)
    pred = (np.linalg.inv(ATA)).dot(ATY)
    return pred
#total least squares method
def tls(x, y):
    #b=w+sqrt(w^2+r^2)/r
    #w=sum(y-ymean)^2-sum(x-xmean)^2
    #r=2*sum*(x-xmean)*(y-ymean)
    #a=ymean-b*xmean
    
    #create U matrix (difference from mean)
    x_tls=x
    y_tls=y
    
    n = len(x_tls)
    
    x_mean=np.mean(x_tls)
    y_mean=np.mean(y_tls)
    
    U=np.vstack(((x_tls-x_mean),(y_tls-y_mean))).T
    # print("U is ", U)
    # print("U shape is ", U.shape)
    
    #create UTU matrix
    UTU=np.dot(U.transpose(),U)
    # print("UTU shape is ", UTU.shape)
    
    #solve for coeffiecients of d=ax+b
    beta=np.dot(UTU.transpose(),UTU)
    
    #get eigenvalues and eigenvectors
    w,v=np.linalg.eig(beta)
    
    #find index of smallest eigenvalue
    index=np.argmin(w)
    #get corresponding eigenvector
    coefficients=v[:,index]
    # print("coefficients are", coefficients)
    # print("coefficients shape is ", coefficients.shape)
    
    a,b=coefficients
    D=a*x_mean+b
    
    tls_value=[]
    for i in range(0,n):
        y_temp=(D-(a*x_tls[i]))/b
        tls_value.append(y_temp)
        
    # print("tls_value ",tls_value )
    
    return tls_value
    
#ransac method
def ransac(x, y):
    A = np.stack((x, np.ones((len(x)), dtype = int)), axis=1)
    #sett a threshold value
    threshold = np.std(y)/3
    ransac_pred = ransac_fit(A, y, 3, threshold)
    ransac_y = A.dot(ransac_pred)
    return ransac_y

def ransac_fit(A, y, sample, threshold):
    #initialize
    num_iter = math.inf
    iter_comp = 0
    max_inlier = 0
    best_fit = None
    prob_outlier = 0
    prob_desired = 0.95 #desired probability
    combined_data = np.column_stack((A,y))
    data_len = len(combined_data)

    #find the number of iterations
    while num_iter > iter_comp:
        #shuffling the rows and taking the first rows
        np.random.shuffle(combined_data)
        sample_data = combined_data[:sample, :]
       
        #estimating the y 
        pred_model = ls_fit(sample_data[:,:-1], sample_data[:, -1:])
      
        #count the inliers 
        y_inliers = A.dot(pred_model)
        err = np.abs(y - y_inliers.T)
       
        #if err is less than the threshold value, then the point is an inlier
        inlier_count = np.count_nonzero(err<threshold)
        print('Inlier Count', inlier_count)
        
        #best fit with maximum inlier count
        if inlier_count > max_inlier:
            max_inlier = inlier_count
            best_fit = pred_model

        #calculating the outlier 
        prob_outlier = 1 - inlier_count/data_len

        #computing the number of iterations 
        num_iter = math.log(1 - prob_desired)/math.log(1-(1-prob_outlier)**sample)
        iter_comp = iter_comp + 1
    return best_fit

y_ls = ls(x, y)
y_tls = tls(x, y)
y_ran = ransac(x, y)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.set_title('eigen_vectors')
ax1.scatter(x,y, label='data points')
ax1.quiver(*origin, *eig_vec1, color=['r'], scale=21)
ax1.quiver(*origin, *eig_vec2, color=['b'], scale=21) 
ax1.set(xlabel='age', ylabel='cost')
ax1.legend()

ax2.set_title('standard least squares')
ax2.scatter(x,y, label='data points')
ax2.plot(x, y_ls, color='red', label='standard least squares model')
ax2.set(xlabel='age', ylabel='cost')
ax2.legend()

ax3.set_title('total least squares')
ax3.scatter(x,y, label='data points')
ax3.plot(x,y_tls, color='blue', label='total least squares model')
ax3.set(xlabel='age', ylabel='cost')
ax3.legend()

ax4.set_title('Ransac')
ax4.plot(x, y_ran, color='yellow', label='ransac model')
ax4.scatter(x,y, label='data points')
ax4.set(xlabel='age', ylabel='cost')
ax4.legend()
plt.show()
