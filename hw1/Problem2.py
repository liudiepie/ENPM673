#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

#input video
d1 = cv2.VideoCapture('ball_video1.mp4')
d2 = cv2.VideoCapture('ball_video2.mp4')
if (d1.isOpened()==False or d2.isOpened()==False):
    print("Error opening video stream or file")

#initialize X and Y
X1 = []
Y1 = []
X2 = []
Y2 = []

#least squares function
def ls(x, y):
    x_square = np.power(x, 2)
    #A matrix quadratic equation y = ax^2 + bx + c
    A = np.stack((x_square, x, np.ones((len(x)), dtype = int)), axis = 1)
    ls_pred = ls_fit(A, y)
    ls_model = A.dot(ls_pred)

    return ls_model

def ls_fit(A, Y):
    A_trans = A.transpose()
    ATA = A_trans.dot(A)
    ATY = A_trans.dot(Y)
    pred = (np.linalg.inv(ATA)).dot(ATY)
    return pred

#read the video frame by frame
while(d1.isOpened() and d2.isOpened()):
    ret1, frame1 = d1.read()
    ret2, frame2 = d2.read()
    if ret1==True and ret2==True:
        #cv2.imshow('F1', frame1)
        #cv2.imshow('F2', frame2)
        
        #find the top and bottom pixel
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        mask1 = cv2.Canny(gray1, 30, 200)
        mask2 = cv2.Canny(gray2, 30, 200)
        contours1, hierarchy1 = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c1 = max(contours1, key = cv2.contourArea)
        c2 = max(contours2, key = cv2.contourArea)
        x1, y1, w1, h1 = cv2.boundingRect(c1)
        x2, y2, w2, h2 = cv2.boundingRect(c2)
        X1.append(x1+w1/2)
        Y1.append(-y1+h1/2)
        X2.append(x2+w2/2)
        Y2.append(-y2+h2/2)
    else:
        break

#A = ((XTX)^-1)XTB
Xdata1 = np.array(X1)
Ydata1 = np.array(Y1)
Xdata2 = np.array(X2)
Ydata2 = np.array(Y2)
#A1 = cv2.solve(X1, Y1, DECOMP_NORMAL)
#A2 = cv2.solve(X2, Y2, DECOMP_NORMAL)
print(X1)
print(Xdata1)
print(Y1)
print(Ydata1)
y1_ls = ls(Xdata1, Ydata1)
y2_ls = ls(Xdata2, Ydata2)
#plot the image
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.set_title('video1')
ax1.scatter(Xdata1, Ydata1, marker='o', color = (0,1,0), label='data points')
ax1.plot(Xdata1, y1_ls, color = 'red', label='least square model')
ax1.set(xlabel='x-axis', ylabel='y-axis')
ax1.legend()

ax2.set_title('video2')
ax2.scatter(Xdata2, Ydata2, marker='o', color = (0,1,0), label='data points')
ax2.plot(Xdata2, y2_ls, color = 'red', label='least square model')
ax2.set(xlabel='x-axis', ylabel='y-axis')
ax2.legend()
plt.show()
if cv2.waitKey(0) & 0xFF == ord('q'):
    #release the video capture object
    d1.release()
    d2.release()

    #close all the frames
    cv2.destroyAllWindows()

