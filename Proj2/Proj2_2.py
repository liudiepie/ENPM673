#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Proj2_2.py
@Time    :   2022/04/01 13:57:01
@Author  :   Cheng Liu 
@Version :   1.0
@Contact :   cliu@umd.edu
@License :   (C)Copyright 2022-2023, Cheng Liu
@Desc    :   None
'''

import numpy as np
import cv2

def filter_colors(image):
	# Filter white pixels
	white_threshold = 200 
	lower_white = np.array([white_threshold, white_threshold, white_threshold])
	upper_white = np.array([255, 255, 255])
	white_mask = cv2.inRange(image, lower_white, upper_white)
	white_image = cv2.bitwise_and(image, image, mask=white_mask)
	# Filter yellow pixels
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([90,100,100])
	upper_yellow = np.array([110,255,255])
	yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
	# Combine the two above images
	image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
	return image2

def region_of_interest(img, vertices):
	#defining a blank mask to start with
	mask = np.zeros_like(img)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
		
	#filling pixels inside the polygon defined by "vertices" with the fill color	
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def bresenham(x1, y1, x2, y2, img):
    dx = x2 - x1
    dy = y2 - y1
    yi = 1
    count = 0
    if dx > 0:
        if dy < 0:
            yi = -1
            dy = -dy
        D = 2*dy - dx
        y = y1
        for x in range(x1, x2+1):
            for i in range(5): #for 5 pixels in x axis, check if they are white
                if img[y-1][x-3+i] == 255:
                    count+=1
            if D > 0:
                y = y + yi
                D = D + 2*(dy-dx)
            else:
                D = D + 2*dy
    else:   #dx < 0
        dx = -dx
        dy = -dy
        yi = 1
        if dy < 0:
            yi = -1
            dy = -dy
        D = 2*dy - dx
        y = y2
        for x in range(x2, x1+1):
            for i in range(5): #for 5 pixels in x axis, check if they are white
                if img[y-1][x-3+i] == 255:
                    count+=1
            if D > 0:
                y = y + yi
                D = D + 2*(dy-dx)
            else:
                D = D + 2*dy
    return count > 415  #if the line has over 415 pixels are white, it will be considered a solid line

def seperate_lines(img, lines, edges):
	# In case of error, don't draw the line(s)
	if lines is None:
		return
	if len(lines) == 0:
		return
	draw_right = True
	draw_left = True
	
	# Find slopes of all lines
	# But only care about lines where abs(slope) > slope_threshold
	slope_threshold = 0.5
	slopes = []
	new_lines = []
	for line in lines:
		x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
		
		# Calculate slope
		if x2 - x1 == 0.:  # corner case, avoiding division by 0
			slope = 999.  # practically infinite slope
		else:
			slope = (y2 - y1) / (x2 - x1)
			
		# Filter lines based on slope
		if abs(slope) > slope_threshold:
			slopes.append(slope)
			new_lines.append(line)
		
	lines = new_lines
	# Split lines into right_lines and left_lines, representing the right and left lane lines
	# Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
	right_lines = []
	left_lines = []
	for i, line in enumerate(lines):
		x1, y1, x2, y2 = line[0]
		img_x_center = img.shape[1] / 2  # x coordinate of center of image
		if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
			right_lines.append(line)
		elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
			left_lines.append(line)		
	# Run linear regression to find best fit line for right and left lane lines
	# Right lane lines
	right_lines_x = []
	right_lines_y = []
	
	for line in right_lines:
		x1, y1, x2, y2 = line[0]
		
		right_lines_x.append(x1)
		right_lines_x.append(x2)
		
		right_lines_y.append(y1)
		right_lines_y.append(y2)
		
	if len(right_lines_x) > 0:
		right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
	else:
		right_m, right_b = 1, 1
		draw_right = False
	# Left lane lines
	left_lines_x = []
	left_lines_y = []
	
	for line in left_lines:
		x1, y1, x2, y2 = line[0]
		
		left_lines_x.append(x1)
		left_lines_x.append(x2)
		
		left_lines_y.append(y1)
		left_lines_y.append(y2)
		
	if len(left_lines_x) > 0:
		left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
	else:
		left_m, left_b = 1, 1
		draw_left = False
	
	# Find 2 end points for right and left lines, used for drawing the line
	# y = m*x + b --> x = (y - b)/m
	y1 = img.shape[0]
	y2 = img.shape[0] * (1 - 0.4)	
	right_x1 = (y1 - right_b) / right_m
	right_x2 = (y2 - right_b) / right_m
	left_x1 = (y1 - left_b) / left_m
	left_x2 = (y2 - left_b) / left_m
	# Convert calculated end points from float to int
	y1 = int(y1)
	y2 = int(y2)
	right_x1 = int(right_x1)
	right_x2 = int(right_x2)
	left_x1 = int(left_x1)
	left_x2 = int(left_x2)
	# Draw the right and left lines on image
	if draw_right:
		draw_lines(img, right_x1, y1, right_x2, y2, edges)
	if draw_left:
		draw_lines(img, left_x1, y1, left_x2, y2, edges)

def draw_lines(img, x1, y1, x2, y2, edges):
    #check whether the line is solid or dashed
    #if lines has over 415 pixels are white
    if bresenham(x1, y1, x2, y2, edges):
        color = [0,255,0] #green for solid
    else: 
        color = [0,0,255] #red for dashed
    #draw line on img
    cv2.line(img, (x1, y1), (x2, y2), color, 3)

def line_detect(img_ori):
    #leave white and yellow pixels
    image = filter_colors(img_ori)
    #gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #clahe
    clahe = cv2.createCLAHE(clipLimit=5)
    img_clahe = clahe.apply(gray)+30
    #gaussian smoothing
    img_blur = cv2.GaussianBlur(img_clahe, (3,3), 0)
    #canny
    edges = cv2.Canny(img_blur, 50, 150)
    # Create masked edges using trapezoid-shaped region-of-interest
    w,h,_ = image.shape
    vertices = np.array([[\
		((h * (1 - 0.85)) // 2, w),\
		((h * (1 - 0.07)) // 2, w - w * 0.4),\
		(h - (h * (1 - 0.07)) // 2, w - w * 0.4),\
		(h - (h * (1 - 0.85)) // 2, w)]]\
		, dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    #HoughLinesP
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 15, np.array([]), minLineLength=10, maxLineGap=20)
    #divide the left and right lines, and draw on original image
    seperate_lines(img_ori, lines, masked_edges)

if __name__ == '__main__':
    #create a videocapture object
    cap = cv2.VideoCapture('whiteline.mp4')
    #create output video
    out = cv2.VideoWriter('output2_2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (960,540))
    #check if camera opened
    if (cap.isOpened()==False):
        print("error")
    #read frame
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            line_detect(frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    
