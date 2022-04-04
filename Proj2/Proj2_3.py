#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Proj2_3.py
@Time    :   2022/04/02 15:14:04
@Author  :   Cheng Liu 
@Version :   1.0
@Contact :   cliu@umd.edu
@License :   (C)Copyright 2022-2023, Cheng Liu
@Desc    :   None
'''

import numpy as np
import cv2
from line import Line, get_lane_lines_img, illustrate_driving_lane

#global initialize
left_line = Line()
right_line = Line()

def filter_colors(image):
	# Filter white pixels
	white_threshold = 200 
	lower_white = np.array([white_threshold, white_threshold, white_threshold])
	upper_white = np.array([255, 255, 255])
	white_mask = cv2.inRange(image, lower_white, upper_white)
	white_image = cv2.bitwise_and(image, image, mask=white_mask)
	# Filter yellow pixels
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([0,80,80])
	upper_yellow = np.array([110,255,255])
	yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
	# Combine the two above images
	image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
	return image2

def perspective_transform(img):
    #warp the image
	img_size = (img.shape[1], img.shape[0])

	src = np.float32(
		[[200, 720],
		[1100, 720],
		[595, 450],
		[685, 450]])
	dst = np.float32(
		[[300, 720],
		[980, 720],
		[300, 0],
		[980, 0]])

	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)

	warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
	unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG

	return warped, unwarped, m, m_inv

def main(image):
	w, h =  image.shape[:2]
	step1 = filter_colors(image)
	step2 = cv2.cvtColor(step1, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe = clahe.apply(step2)
	step3 = cv2.GaussianBlur(clahe, (3,3), 0)
	(T, step4) = cv2.threshold(step3, 100, 255, cv2.THRESH_BINARY)
	step5, _, m, minv = perspective_transform(step4)	
	step6 = get_lane_lines_img(step5, left_line, right_line)
	step7_1, step7_2 = illustrate_driving_lane(step6, left_line, right_line)
	step8 = cv2.warpPerspective(step7_2, minv, (h, w))
	result = cv2.addWeighted(image, 1, step8, 1, 0)
	return result

if __name__ == '__main__':
    #create a videocapture object
    cap = cv2.VideoCapture('challenge.mp4')
    #create output video
    out = cv2.VideoWriter('output2_3.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (1280,720))
    #check if camera opened
    if (cap.isOpened()==False):
        print("error")
    #read frame
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            result = main(frame)
            out.write(result)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()