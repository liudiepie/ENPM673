#!/usr/bin/env python
# -*-coding:utf-8 -*-
import numpy as np
import cv2

class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # Set the width of the windows +/- margin
        self.window_margin = 56
        # x values of the fitted line over the last n iterations
        self.prevx = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        # starting x_value
        self.startx = None
        # ending x_value
        self.endx = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # road information
        self.road_info = None
        self.curvature = None
        self.deviation = None

def measure_curvature(left_lane, right_lane):
    ploty = left_lane.ally

    leftx, rightx = left_lane.allx, right_lane.allx

    leftx = leftx[::-1]     # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]   # Reverse to match top-to-bottom in y

    # Define y-value where we want radius of curvature
    # choose the maximum y-value, corresponding to the bottom of the image    
    y_eval = np.max(ploty)
    
    # Below is the calculation of radius of curvature after correcting for scale in x and y
    # Define conversions in x and y from pixels space to meters
    lane_width = abs(right_lane.startx - left_lane.startx)
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7*(720/1280) / lane_width  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    
    # Calculate the new radius of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    
    # radius of curvature result
    left_lane.radius_of_curvature = left_curverad
    right_lane.radius_of_curvature = right_curverad


def smoothing(lines, prev_n_lines=3):
    # collect lines & print average line
    lines = np.squeeze(lines)       
    avg_line = np.zeros((720))

    for i, line in enumerate(reversed(lines)):
        if i == prev_n_lines:
            break
        avg_line += line
    avg_line = avg_line / prev_n_lines

    return avg_line


def line_search_reset(binary_img, left_lane, right_line):
    #histogram along all the columns in the lower half of the image
    histogram = np.sum(binary_img[int(binary_img.shape[0] / 2):, :], axis=0)

    #create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

    #find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0] / 2)
    leftX_base = np.argmax(histogram[:midpoint])
    rightX_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    num_windows = 9
    
    # Set height of windows
    window_height = np.int(binary_img.shape[0] / num_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    current_leftX = leftX_base
    current_rightX = rightX_base

    # Set minimum number of pixels found to recenter window
    min_num_pixel = 50

    # Create empty lists to receive left and right lane pixel indices
    win_left_lane = []
    win_right_lane = []

    window_margin = left_lane.window_margin

    # Step through the windows one by one
    for window in range(num_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_img.shape[0] - (window + 1) * window_height
        win_y_high = binary_img.shape[0] - window * window_height
        win_leftx_min = current_leftX - window_margin
        win_leftx_max = current_leftX + window_margin
        win_rightx_min = current_rightX - window_margin
        win_rightx_max = current_rightX + window_margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
            nonzerox <= win_leftx_max)).nonzero()[0]
        right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
            nonzerox <= win_rightx_max)).nonzero()[0]
        # Append these indices to the lists
        win_left_lane.append(left_window_inds)
        win_right_lane.append(right_window_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(left_window_inds) > min_num_pixel:
            current_leftX = np.int(np.mean(nonzerox[left_window_inds]))
        if len(right_window_inds) > min_num_pixel:
            current_rightX = np.int(np.mean(nonzerox[right_window_inds]))

    # Concatenate the arrays of indices
    win_left_lane = np.concatenate(win_left_lane)
    win_right_lane = np.concatenate(win_right_lane)

    # Extract left and right line pixel positions
    leftx= nonzerox[win_left_lane]
    lefty =  nonzeroy[win_left_lane]
    rightx = nonzerox[win_right_lane]
    righty = nonzeroy[win_right_lane]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_lane.current_fit = left_fit
    right_line.current_fit = right_fit

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_lane.prevx.append(left_plotx)
    right_line.prevx.append(right_plotx)

    if len(left_lane.prevx) > 10:
        left_avg_line = smoothing(left_lane.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_lane.current_fit = left_avg_fit
        left_lane.allx, left_lane.ally = left_fit_plotx, ploty
    else:
        left_lane.current_fit = left_fit
        left_lane.allx, left_lane.ally = left_plotx, ploty

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plotx, ploty

    left_lane.startx, right_line.startx = left_lane.allx[len(left_lane.allx)-1], right_line.allx[len(right_line.allx)-1]
    left_lane.endx, right_line.endx = left_lane.allx[0], right_line.allx[0]

    # Set detected=True for both lines
    left_lane.detected, right_line.detected = True, True
    
    measure_curvature(left_lane, right_line)
    
    return out_img

def line_search_tracking(b_img, left_line, right_line):
    #it is based on the lines which detected in the previous frame
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((b_img, b_img, b_img)) * 255

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Get margin of windows from Line class. Adjust this number.
    window_margin = left_line.window_margin

    left_line_fit = left_line.current_fit
    right_line_fit = right_line.current_fit
    leftx_min = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
    leftx_max = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
    rightx_min = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
    rightx_max = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin

    # Identify the nonzero pixels in x and y within the window
    left_inds = ((nonzerox >= leftx_min) & (nonzerox <= leftx_max)).nonzero()[0]
    right_inds = ((nonzerox >= rightx_min) & (nonzerox <= rightx_max)).nonzero()[0]

    # Extract left and right line pixel positions
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    leftx_avg = np.average(left_plotx)
    rightx_avg = np.average(right_plotx)

    left_line.prevx.append(left_plotx)
    right_line.prevx.append(right_plotx)

    if len(left_line.prevx) > 10:  # take at least 10 previously detected lane lines for reliable average
        left_avg_line = smoothing(left_line.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plotx, ploty
    else:
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = left_plotx, ploty

    if len(right_line.prevx) > 10: # take at least 10 previously detected lane lines for reliable average
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plotx, ploty

    # Compute Standard Deviation of the distance between X positions of pixels of left and right lines
    # If this STDDEV is too high, then we need to reset our line search, using line_search_reset
    stddev = np.std(right_line.allx - left_line.allx)

    if (stddev > 80):
        left_line.detected = False

    left_line.startx, right_line.startx = left_line.allx[len(left_line.allx) - 1], right_line.allx[len(right_line.allx) - 1]
    left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

    measure_curvature(left_line, right_line)
    
    return out_img


def get_lane_lines_img(binary_img, left_line, right_line):
    #check if the line detected before
    if left_line.detected == False:
        return line_search_reset(binary_img, left_line, right_line)
    else:
        return line_search_tracking(binary_img, left_line, right_line)


def illustrate_driving_lane(img, left_line, right_line, lane_color=(0, 255, 255), road_color=(0, 0, 255)):
    # Create an empty image to draw on
    window_img = np.zeros_like(img)

    window_margin = left_line.window_margin
    left_plotx, right_plotx = left_line.allx, right_line.allx
    ploty = left_line.ally
    mid_plotx = (left_line.allx + right_line.allx)//2

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_plotx - window_margin/5, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_plotx + window_margin/5, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_plotx - window_margin/5, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_plotx + window_margin/5, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    mid_line = np.array([np.transpose(np.vstack([mid_plotx, ploty]))])
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), lane_color)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), lane_color)


    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_plotx+window_margin/5, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx-window_margin/5, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), road_color)

    # Draw arrows onto the warped blank image
    for i in range(0, len(mid_line[0])-1, 36):
        if (i/36)%2 == 0:
            cv2.arrowedLine(window_img, np.int_(mid_line[0][i+35]), np.int_(mid_line[0][i]), (255,0,0), 9, tipLength=0.5)
    result = cv2.addWeighted(img, 1, window_img, 0.3, 0)

    return result, window_img


def get_measurements(left_line, right_line):
    # take average of radius of left curvature and right curvature 
    curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2

    # calculate direction using X coordinates of left and right lanes 
    direction = ((left_line.endx - left_line.startx) + (right_line.endx - right_line.startx)) / 2
     
    if curvature > 2000 and abs(direction) < 100:
        road_info = 'Straight'
        curvature = -1
    elif curvature <= 2000 and direction < - 50:
        road_info = 'curving to Left'
    elif curvature <= 2000 and direction > 50:
        road_info = 'curving to Right'
    else:
        if left_line.road_info != None:
            road_info = left_line.road_info
            curvature = left_line.curvature
        else:
            road_info = 'None'
            curvature = curvature

    center_lane = (right_line.startx + left_line.startx) / 2
    lane_width = right_line.startx - left_line.startx

   
    center_car = 720 / 2
    if center_lane > center_car:
        deviation = str(round(abs(center_lane - center_car), 3)) + 'm Left'
    elif center_lane < center_car:
        deviation = str(round(abs(center_lane - center_car), 3)) + 'm Right'
    else:
        deviation = 'by 0 (Centered)'

    left_line.road_info = road_info
    left_line.curvature = curvature
    left_line.deviation = deviation

    return road_info, curvature, deviation

