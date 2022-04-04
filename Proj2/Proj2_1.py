#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Proj2_1.py
@Time    :   2022/03/31 22:44:04
@Author  :   Cheng Liu 
@Version :   1.0
@Contact :   cliu@umd.edu
@License :   (C)Copyright 2022-2023, Cheng Liu
@Desc    :   None
'''

import numpy as np
import cv2
import glob


path = sorted(glob.glob("adaptive_hist_data/*.png"))
images = np.empty(len(path), dtype=object)
images_his = np.empty(len(path), dtype=object)
images_ahe = np.empty(len(path), dtype=object)
out1 = cv2.VideoWriter('out1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (1224,370), 0)
out2 = cv2.VideoWriter('out2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (1224,370), 0)

def ahe(img_ori):
    img = img_ori.copy()
    for r in range(0, img.shape[0], 100):
        for c in range(0, img.shape[1], 100):
            img[r:r+100, c:c+100] = histrogram(img[r:r+100, c:c+100])
    return img

def histrogram(img_ori):
    img = img_ori.copy()
    a = np.zeros((256,),dtype=np.float16)
    b = np.zeros((256,),dtype=np.float16)

    height,width=img.shape

    #finding histogram
    for i in range(width):
        for j in range(height):
            g = img[j,i]
            a[g] = a[g]+1
    #print(a)

    #performing histogram equalization
    tmp = 1.0/(height*width)
    b = np.zeros((256,),dtype=np.float16)
    
    for i in range(256):
        for j in range(i+1):
            b[i] += a[j] * tmp
        b[i] = round(b[i] * 255)
    
    # b now contains the equalized histogram
    b=b.astype(np.uint8)
    #print(b)

    #Re-map values from equalized histogram into the image
    for i in range(width):
        for j in range(height):
            g = img[j,i]
            img[j,i]= b[g]
    return img

def main():

    for img in path:
        img = cv2.imread(img,0)
        images_his = histrogram(img)
        images_ahe = ahe(img)
        out1.write(images_his)
        out2.write(images_ahe)
        print("finish 1 image")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("Done!")
    out1.release()
    out2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
