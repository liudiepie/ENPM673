#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   Proj3.py
@Time    :   2022/04/13 22:12:03
@Author  :   Cheng Liu 
@Version :   1.0
@Contact :   cliu@umd.edu
@License :   (C)Copyright 2022-2023, Cheng Liu
@Desc    :   None
'''
import cv2
import numpy as np
from tqdm import tqdm


def normalize(uv):
    
    uv_dash = np.mean(uv, axis=0)
    u_dash ,v_dash = uv_dash[0], uv_dash[1]

    u_cap = uv[:,0] - u_dash
    v_cap = uv[:,1] - v_dash

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_dash],[0,1,-v_dash],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T

def estimate_fundamental(features):
    normalized = True

    x1 = features[:,0:2]
    x2 = features[:,2:4]

    if x1.shape[0] > 7:
        if normalized == True:
            x1_norm, T1 = normalize(x1)
            x2_norm, T2 = normalize(x2)
        else:
            x1_norm,x2_norm = x1,x2
            
        A = np.zeros((len(x1_norm),9))
        for i in range(0, len(x1_norm)):
            x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        ###---Using SVD for solving for Fundamental Matrix---###
        U, S, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1]
        F = F.reshape(3,3)

        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0
        F = np.dot(u, np.dot(s, vt))

        if normalized:
            F = np.dot(T2.T, np.dot(F, T1))
        return F

    else:
        return None

def error_fundamental(feature, F): 
    x1,x2 = feature[0:2], feature[2:4]
    x1_tmp=np.array([x1[0], x1[1], 1]).T
    x2_tmp=np.array([x2[0], x2[1], 1])

    error = np.dot(x1_tmp, np.dot(F, x2_tmp))
    
    return np.abs(error)

def RANSAC(features):
    n_iterations = 1000
    error_thresh = 0.02
    inliers_thresh = 0
    chosen_indices = []
    F_matrix = 0

    for i in range(0, n_iterations):
        indices = []
        #select 8 points randomly
        n_rows = features.shape[0]
        random_indices = np.random.choice(n_rows, size=8)
        features_8 = features[random_indices, :] 
        f_8 = estimate_fundamental(features_8)
        for j in range(n_rows):
            feature = features[j]
            error = error_fundamental(feature, f_8)
            if error < error_thresh:
                indices.append(j)

        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            chosen_indices = indices
            F_matrix = f_8

    filtered_features = features[chosen_indices, :]
    return F_matrix, filtered_features

def F2E(F, K):

    E = np.dot(K.T, np.dot(F, K))
    U, S, V_T = np.linalg.svd(E)

    E = np.dot(U, np.dot(np.diag([1, 1, 0]), V_T))
    return E

def E2RT(E):
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    T = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    T.append(U[:, 2])
    T.append(-U[:, 2])
    T.append(U[:, 2])
    T.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            T[i] = -T[i]
    return R, T

def matchImageSizes(imgs):
    images = imgs.copy()
    sizes = []
    for image in images:
        x, y, ch = image.shape
        sizes.append([x, y, ch])

    sizes = np.array(sizes)
    x_target, y_target, _ = np.max(sizes, axis = 0)
    
    images_resized = []

    for i, image in enumerate(images):
        image_resized = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
        image_resized[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
        images_resized.append(image_resized)

    return images_resized

def epipolar_lines(img1,img2,F,pts1,pts2, rectified):
    lines1, lines2 = [], []
    img_epi1 = img1.copy()
    img_epi2 = img2.copy()

    for i in range(pts1.shape[0]):
        x1 = np.array([pts1[i,0], pts1[i,1], 1]).reshape(3,1)
        x2 = np.array([pts2[i,0], pts2[i,1], 1]).reshape(3,1)

        line2 = np.dot(F, x1)
        lines2.append(line2)

        line1 = np.dot(F.T, x2)
        lines1.append(line1)

        #solve for in and max x values based on equation of line
        if not rectified:
            y2_min = 0
            y2_max = img2.shape[0]
            x2_min = -(line2[1]*y2_min + line2[2])/line2[0]
            x2_max = -(line2[1]*y2_max + line2[2])/line2[0]

            y1_min = 0
            y1_max = img1.shape[0]
            x1_min = -(line1[1]*y1_min + line1[2])/line1[0]
            x1_max = -(line1[1]*y1_max + line1[2])/line1[0]
        else:
            x2_min = 0
            x2_max = img2.shape[1] - 1
            y2_min = -line2[2]/line2[1]
            y2_max = -line2[2]/line2[1]

            x1_min = 0
            x1_max = img1.shape[1] -1
            y1_min = -line1[2]/line1[1]
            y1_max = -line1[2]/line1[1]

        #draw circles on images for points connecting epipolar lines
        cv2.circle(img_epi2, (int(pts2[i,0]),int(pts2[i,1])), 10, (0,0,255), -1)
        img_epi2 = cv2.line(img_epi2, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255, 0, int(i*2.55)), 2)
    

        cv2.circle(img_epi1, (int(pts1[i,0]),int(pts1[i,1])), 10, (0,0,255), -1)
        img_epi1 = cv2.line(img_epi1, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (255, 0, int(i*2.55)), 2)

    #Resize images and concatenate back together
    image_1, image_2 = matchImageSizes([img_epi1, img_epi2])
    concat = np.concatenate((image_1, image_2), axis = 1)
    concat = cv2.resize(concat, (1920, 660))
    return lines1, lines2, concat

def SSD(gray1, gray2, ndisp):
    kernel_size = 5
    h, w = gray1.shape
    Disparity = np.zeros([h, w])
    indices = np.arange(0, ndisp). reshape(-1,1)
    indices_j = np.clip(indices+np.arange(0, w-kernel_size+1), 0, w-kernel_size).T
    indices_i = np.arange(indices_j.shape[0]).reshape(-1,1)

    for i in tqdm(range(kernel_size//2, h-kernel_size//2)):
        kernels1 = []
        kernels2 = []
        for j in range(kernel_size//2, w-kernel_size//2):
            kernel1 = gray1[i-kernel_size//2:i+kernel_size//2+1, j-kernel_size//2:j+kernel_size//2+1]
            kernel2 = gray2[i-kernel_size//2:i+kernel_size//2+1, j-kernel_size//2:j+kernel_size//2+1]
            kernels1.append(kernel1.reshape(-1))
            kernels2.append(kernel2.reshape(-1))
        kernels1 = np.array(kernels1).astype(float)
        kernels2 = np.array(kernels2).astype(float)
        SSD = (kernels1**2).sum(-1).reshape(-1,1) + (kernels2**2).sum(-1).reshape(1,-1) - 2*(kernels1 @ kernels2.T)
        SSD = SSD[indices_i, indices_j]
        n = kernels1.shape[0]
        disparity = SSD.argmin(-1)
        Disparity[i,kernel_size//2:kernel_size//2+n] = disparity
    Disparity = np.maximum(Disparity, 0)
    Disparity = Disparity/Disparity.max()*255
    return Disparity

def Calibration(img1, img2, K):
    #apply sift
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    #find match point
    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x :x.distance)
    chosen_matches = matches[0:100]
    matched_pairs = []
    for i, m1 in enumerate(chosen_matches):
        pt1 = kp1[m1.queryIdx].pt
        pt2 = kp2[m1.trainIdx].pt
        matched_pairs.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    matched_pairs = np.array(matched_pairs).reshape(-1, 4)
    #apply ransac to find the best fundamental matrix
    F, inliers = RANSAC(matched_pairs)
    set1, set2 = inliers[:,0:2], inliers[:,2:4]
    #get essential matrix
    E = F2E(F, K)
    #get rotation matrix and translation matrix
    R, T = E2RT(E)
    print("Fundamental matrix: ", F)
    print("Rotation matrix: ", R)
    print("Translation matrix: ", T)
    return set1, set2, F

def Rectification(img1, img2, pts1, pts2, F):
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    #get H1 and H2 of left and right image
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))
    #warp the image to rectify
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
    #transform the featrue point with H1 and H2
    pts1_rectified = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), H1).reshape(-1,2)
    pts2_rectified = cv2.perspectiveTransform(pts2.reshape(-1, 1, 2), H2).reshape(-1,2)
    H2_T_inv =  np.linalg.inv(H2.T)
    H1_inv = np.linalg.inv(H1)
    F_rectified = np.dot(H2_T_inv, np.dot(F, H1_inv))
    print("H1 for left: ", H1)
    print("H1 for right: ", H2)
    lines1, lines2, unrectified = epipolar_lines(img1, img2, F, pts1, pts2, False)
    lines1_rectified, lines2_rectified, rectified = epipolar_lines(img1_rectified, img2_rectified, F_rectified, pts1_rectified, pts2_rectified, True)
    cv2.imwrite("rectified.png", rectified)
    cv2.imwrite("unrectified.png", unrectified)
    return img1_rectified, img2_rectified

def Correspondence(gray1_rect, gray2_rect, ndisp):
    #apply SSD to get disparity 
    disp_img = SSD(gray1_rect, gray2_rect, ndisp)
    cv2.imwrite("disparity.png", disp_img)
    disp_img = np.array(disp_img).astype(np.uint8)
    #utilize colormap to get heatmap
    heatmap = cv2.applyColorMap(disp_img, cv2.COLORMAP_HOT)
    cv2.imwrite("heat.png", heatmap)
    return disp_img

def ComputeDepthImage(shape, disp_img, K, baseline):
    f = K[0,0]
    depth = np.zeros(shape=shape).astype(float)
    #apply focal * baseline / disparity to get depth
    depth[disp_img > 0] = (f * baseline) / (disp_img[disp_img > 0])
    img_depth = ((depth/depth.max())*255).astype(np.uint8)
    #utilize colormap to get heatmap
    heatmap = cv2.applyColorMap(img_depth, cv2.COLORMAP_HOT)
    cv2.imwrite("depth.png", img_depth)
    cv2.imwrite("heatdepth.png", heatmap)


def main():
    #default image
    img1 = cv2.imread('data/curule/im0.png')
    img2 = cv2.imread('data/curule/im1.png')
    K = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0 ,1]])
    baseline = 88.39
    ndisp = 220
    image_type = input("Choose input image, curule:1, octagon:2, pendulum:3 ")
    if image_type == '1':
        img1 = cv2.imread('data/curule/im0.png')
        img2 = cv2.imread('data/curule/im1.png')
        K = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0 ,1]])
        baseline = 88.39
        ndisp = 220
        print("start processing curule image")
    elif image_type == '2':
        img1 = cv2.imread('data/octagon/im0.png')
        img2 = cv2.imread('data/octagon/im1.png')
        K = np.array([[1742.11, 0, 804.90], [0,1742.11, 541.22], [0, 0, 1]])
        baseline = 221.76    
        ndisp = 100
        print("start processing octagon image")
    elif image_type == '3':
        img1 = cv2.imread('data/pendulum/im0.png')
        img2 = cv2.imread('data/pendulum/im1.png')
        K = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
        baseline = 537.75   
        ndisp = 180
        print("start processing pendulum image")
    else:
        print("not valid input, processing default")
    gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #Step1 calibration
    set1, set2, F = Calibration(gray1, gray2, K)
    #Step2 rectification
    img1_rect, img2_rect = Rectification(img1, img2, set1, set2, F)
    #Step3 correspondence
    gray1_rect = cv2.cvtColor(img1_rect,cv2.COLOR_BGR2GRAY)
    gray2_rect = cv2.cvtColor(img2_rect,cv2.COLOR_BGR2GRAY)
    disp_img = Correspondence(gray1_rect, gray2_rect, ndisp)
    #Step4 compute depth image
    ComputeDepthImage(gray1.shape, disp_img, K, baseline)
    

if __name__ == '__main__':
    main()