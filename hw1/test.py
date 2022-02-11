#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import random
import cv2

#input video
d1 = cv2.VideoCapture('ball_video1.mp4')
d2 = cv2.VideoCapture('ball_video2.mp4')
if (d1.isOpened()==False or d2.isOpened()==False):
    print("Error opening video stream or file")

#read the video frame by frame
while(d1.isOpened() and d2.isOpened()):
    ret1, frame1 = d1.read()
    if ret1==True:
        #cv2.imshow('F1', frame1)
        #cv2.imshow('F2', frame2)
        
        #find the top and bottom pixel
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        mask1 = cv2.Canny(gray, 30, 200)
        #mask1 = cv2.inRange(frame1, lower_red, upper_red)
        contours1, hierarchy1 = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours1:
            continue
        c1 = max(contours1, key = cv2.contourArea)
        x1, y1, w1, h1 = cv2.boundingRect(c1)
        print("number of contours found" + str(len(contours1)))

        cv2.drawContours(frame1, contours1, -1, (0, 255, 0), 3)
        cv2.resize(frame1, (200, 100))
        cv2.imshow('Contours', frame1)
        cv2.waitKey(0)
    else:
        break


if cv2.waitKey(0) & 0xFF == ord('q'):
    #release the video capture object
    d1.release()
    d2.release()

    #close all the frames
    cv2.destroyAllWindows()

