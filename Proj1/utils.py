import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import fftpack


def AR_Code_detection(img):
    #scipy fft method
    '''
    im_fft = fftpack.fft2(img)
    keep_fraction = 0.1
    im_fft2 = im_fft.copy()
    r, c = im_fft2.shape
    im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    im_new = fftpack.ifft2(im_fft2).real
    '''
    #numpy fft method
    dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
    im_fft = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(im_fft[:,:,0], im_fft[:,:,1]))
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 80
    center = [crow, ccol]
    x,y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
    mask[mask_area] = 0
    fshift = im_fft*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    kernel = np.ones((30,30), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    closing = np.array(closing, dtype=np.uint8)
    edges = cv2.Canny(closing, 200, 250)
    corners = cv2.goodFeaturesToTrack(img_back, 25 ,0.01,50)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img_back, (x,y), 3, 255, 10)
    return img_back, corners, magnitude_spectrum

def plot_spectrum(im_fft):
    #plot spectrum of the image
    from matplotlib.colors import LogNorm
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

def Decode_custom_AR_tag(frame):
    #check the direction of the image
    dim = frame.shape[0]
    april_img = np.zeros((dim,dim,3), np.uint8)
    grid_size = 8
    k = dim//grid_size
    sx = 0
    sy = 0
    font = cv2.FONT_HERSHEY_TRIPLEX
    decode = np.zeros((grid_size,grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            roi = frame[sy:sy+k, sx:sx+k]
            if roi.mean() > 255//2:  #if white, code as a "1"
                decode[i][j] = 1
                cv2.rectangle(april_img,(sx,sy),(sx+k,sy+k),(255,255,255),-1)
            cv2.rectangle(april_img,(sx,sy),(sx+k,sy+k),(127,127,127),1)
            sx += k
        sx = 0
        sy += k
    # Id of April Tag is contained in the inner four elements of the 8x8 tag
    # a  b
    # d  c
    a = str(int(decode[3][3]))
    b = str(int(decode[3][4]))
    c = str(int(decode[4][4]))
    d = str(int(decode[4][3]))

    #Add the binary value as text to the appropriate cell
    cv2.putText(april_img,a,(3*k+int(k*.3),3*k+int(k*.7)),font,.6,(227,144,27),2)
    cv2.putText(april_img,b,(4*k+int(k*.3),3*k+int(k*.7)),font,.6,(227,144,27),2)
    cv2.putText(april_img,d,(3*k+int(k*.3),4*k+int(k*.7)),font,.6,(227,144,27),2)
    cv2.putText(april_img,c,(4*k+int(k*.3),4*k+int(k*.7)),font,.6,(227,144,27),2)

    #Determine orientation of image to properly add up binary values
    if decode[5,5] == 1: #Bottom-right (BR) corner is at TR
        orientation = 3
        id_binary = a+b+c+d
        center = (5*k+(k//2),5*k+(k//2))
        cv2.circle(april_img,center,k//4,(0,0,255),-1)
    elif decode[2,5] == 1: #Bottom-right (BR) corner is at TL
        orientation = 2
        id_binary = d+a+b+c
        center = (5*k+(k//2),2*k+(k//2))
        cv2.circle(april_img,center,k//4,(0,0,255),-1)
    elif decode[2,2] == 1: #Bottom-right (BR) corner is at BL
        orientation = 1
        id_binary = c+d+a+b
        center = (2*k+(k//2),2*k+(k//2))
        cv2.circle(april_img,center,k//4,(0,0,255),-1)
    elif decode[5,2] == 1: #Bottom-right (BR) corner is at BR
        orientation = 0
        id_binary = b+c+d+a
        center = (2*k+(k//2),5*k+(k//2))
        cv2.circle(april_img,center,k//4,(0,0,255),-1)
    else:  #Just in case used on different image
        orientation = 0
        id_binary = '0000'
    return april_img,id_binary,orientation

def markUpImageCorners(image):
    #Get corners
    marked_corners=image.copy()
    
    #Get tag corners using Shi-Tomasi
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_img, 30, 0.1, 10)
    corners = np.int0(corners)
    
    #Draw circles on corners
    for i in corners:
        x, y = i.ravel()
        cv2.circle(marked_corners, (x, y), 3, (0, 0, 255), -1)
    
    return marked_corners

def rotateTestudo(image, orientation):
    #rotate the direction of testudo
    if orientation == 1:
        new_img = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 2:
        new_img = cv2.rotate(image,cv2.ROTATE_180)
    elif orientation == 3:
        new_img = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        new_img = image
    return new_img


def find_tag(contours):
    #find the corners of April tag
    hull = cv2.convexHull(contours)
    outer = []
    contours_new = []
    In_array = False
    #append unqualified points
    for i in range(len(contours)):
        for h in hull:
            if contours[i][0][0] == h[0][0] and contours[i][0][1] == h[0][1]:
                outer.append(i)
            if contours[i][0][0] == 0 or contours[i][0][1] == 0:
                outer.append(i)
            if contours[i][0][0] == np.inf or contours[i][0][1] == np.inf:
                outer.append(i)
    #append corners to new array and remove the unsuitable points
    for i in range(len(contours)):
        In_array = i in outer
        if In_array:
            continue
        contours_new.append(contours[i])
    contours_new = np.array(contours_new)
    hull_inside = cv2.convexHull(contours_new)
    corners=[]
    top = [0,0]
    bottom = [0, np.inf]
    left = [np.inf, 0]
    right = [0,0]
    #check the sequence of points
    for i in hull_inside:
        if top[1]< i[0][1]:
            top = i[0]
        if bottom[1]>i[0][1]:
            bottom = i[0]
        if left[0]> i[0][0]:
            left = i[0]
        if right[0]< i[0][0]:
            right = i[0]
    corners.append([left, bottom, right, top])
    return corners

def find_points(frame, contour):
    #find poly points and draw the contour
    height = frame.shape[0]
    length = frame.shape[1]
    matrix = np.zeros((height,length),dtype=np.int32)
    cv2.drawContours(matrix, contour, -1, (1), thickness=-1)
    indexes = np.nonzero(matrix)
    poly_points = len(indexes[0])
    return poly_points

def solveHomography(tag, dim):
    #homography with svd
    xp = np.array([0,dim,dim,0])
    yp = np.array([0,0,dim,dim])
    x = np.array([tag[0][0][0], tag[0][1][0], tag[0][2][0], tag[0][3][0]])
    y = np.array([tag[0][0][1], tag[0][1][1], tag[0][2][1], tag[0][3][1]])
    A = np.matrix([[-x[0], -y[0], -1, 0, 0, 0, x[0]*xp[0], y[0]*xp[0], xp[0]],
                    [0, 0, 0, -x[0], -y[0], -1, x[0]*yp[0], y[0]*yp[0], yp[0]],
                    [-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]],
                    [0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]],
                    [-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]],
                    [0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]], 
                    [-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]],
                    [0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]]])
    U, S, Vh = np.linalg.svd(A)
    l = Vh[-1,:]/Vh[-1,-1]
    H = np.reshape(l, (3,3))
    return H

def warp(H, image, h, w):
    #index and linearize
    ind_y, ind_x = np.indices((h, w), dtype=np.float32)
    index_linearized = np.array([ind_x.ravel(), ind_y.ravel(), np.ones_like(ind_x).ravel()])
    
    #warp
    map_ind = H.dot(index_linearized)
    map_x, map_y = map_ind[:-1]/map_ind[-1]
    map_x = map_x.reshape(h,w).astype(np.float32)
    map_y = map_y.reshape(h,w).astype(np.float32)

    #new image
    warped_img = np.zeros((h,w,3),dtype="uint8")
    map_x[map_x>=image.shape[1]] = -1
    map_x[map_x<0] = -1
    map_y[map_y>=image.shape[0]] = -1
    map_y[map_y<0] = -1

    for new_x in range(w):
        for new_y in range(h):
            x = int(map_x[new_y, new_x])
            y = int(map_y[new_y, new_x])

            if x == -1 or y == -1:
                pass
            else:
                warped_img[new_y,new_x] = image[y,x]
    return warped_img
    
def solveProjectionMatrix(K , H):
    #calculate rotation and translation
    H=H*(-1)
    Bhat=np.dot(np.linalg.inv(K),H)
    #Ensure positive Bhat
    if np.linalg.norm(Bhat)>0:
        B=1*Bhat
    else:
        B=-1*Bhat
    
    b_1=B[:,0]
    b_2=B[:,1]
    b_3=B[:,2]
    
    #lambda is average length of the first two columns of B
    lambda_=np.sqrt(np.linalg.norm(b_1,2)* np.linalg.norm(b_2,2))
    
    #normalize vectors
    rot_1=b_1/ lambda_
    rot_2=b_2/ lambda_
    trans=b_3/ lambda_
    
    c=rot_1+rot_2
    p=np.cross(rot_1,rot_2)
    d=np.cross(c,p)
    
    #orthogonal basis
    rot_1=np.dot(c/ np.linalg.norm(c,2) + d / np.linalg.norm(d,2), 1 / np.sqrt(2))
    rot_2=np.dot(c/ np.linalg.norm(c,2) - d / np.linalg.norm(d,2), 1 / np.sqrt(2))
    rot_3=np.cross(rot_1,rot_2)
    
    R_t=np.stack((rot_1,rot_2,rot_3,trans)).T
    
    #P = K * [R | t]
    projectionMatrix=np.dot(K,R_t)
    
    return projectionMatrix

#solves for points directly above tag to build a cube from
def projectionPoints(corners, P):
    projected_corners=[]
    #Separate corners of AR tag into x, y, z coordinates
    x = []
    y = []
    z = []
  
    for point in corners:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2]) #dimensions of cube in -z direction

    #points shifted in world frame
      # X_w= [x1, x2,x3, x4],
    #      [y1, y2,y3, y4],
    #      [z1, z2,z3, z4],
    #      [1,  1,  1,  1]
    
    X_w=np.stack((np.array(x),np.array(y),np.array(z),np.ones(len(x))))
    # print("skewed camera top corner points ",X_w)
    
    #use Projection Matrix to shift back to camera frame
    sX_c2=np.dot(P,X_w)

    #camera frame homography
    X_c2=sX_c2/sX_c2[2,:]
    # print("X_c2 is: ", X_c2)
    
    
    for i in range(4):
        projected_corners.append([int(X_c2[0][i]),int(X_c2[1][i])])
        
    return projected_corners

def solveHomographyCube(AR_corners, top_corners):
    #Define the eight points to compute the homography matrix
    x=[]
    y=[]
    xp=[]
    yp=[]
    
    #convert corners into x and y coordinates
    for point in AR_corners:
        x.append(point[0])
        y.append(point[1])
        
    for point in top_corners:
        xp.append(point[0])
        yp.append(point[1])

    #make A an 8x9 matrix
    n = 9 #9 columns
    m = 8 #8 rows
    A = np.empty([m, n])
    
    #A matrix is:
    # Even rows (0,2,4,6): [[-x, -y, -1,0,0,0, x*x', y*x', x'],
    # Odd rows (1,3,5,7): [0,0,0, -x, -y, -1, x*y', y*y', y']]

    val = 0
    for row in range(0,m):
        if (row%2) == 0: #Even rows
            A[row,0] = -x[val]
            A[row,1] = -y[val]
            A[row,2] = -1
            A[row,3] = 0
            A[row,4] = 0
            A[row,5] = 0
            A[row,6] = x[val]*xp[val]
            A[row,7] = y[val]*xp[val]
            A[row,8] = xp[val]

        else: #odd rows
            A[row,0] = 0
            A[row,1] = 0
            A[row,2] = 0
            A[row,3] = -x[val]
            A[row,4] = -y[val]
            A[row,5] = -1
            A[row,6] = x[val]*yp[val]
            A[row,7] = y[val]*yp[val]
            A[row,8] = yp[val]
            val += 1

    #Conduct SVD to get V
    U,S,V = np.linalg.svd(A)
    
    #Find the eigenvector column of V that corresponds to smallest value (last column)
    x=V[-1]

    # reshape x into 3x3 matrix to have H
    H = np.reshape(x,[3,3])

    return H

def connectCubeCornerstoTag(AR_corners,cube_corners):
    lines = []
    #point 1 (i=0): (0,0), (1,0), (0,1), (1,1)
    
    for i in range(len(AR_corners)):
        if i==3: #last corner
            p1 = AR_corners[i]
            p2 = AR_corners[0]
            p3 = cube_corners[0]
            p4 = cube_corners[i]
        else:
            p1 = AR_corners[i]
            p2 = AR_corners[i+1]
            p3 = cube_corners[i+1]
            p4 = cube_corners[i]
            
         #build array of connecting lines   
        lines.append(np.array([p1,p2,p3,p4], dtype=np.int32))
        # print("Current contours ", lines[i])
        
        #append tag corners and top square corners
    lines.append(np.array([AR_corners[0],AR_corners[1],AR_corners[2],AR_corners[3]], dtype=np.int32))
    lines.append(np.array([cube_corners[0],cube_corners[1],cube_corners[2],cube_corners[3]], dtype=np.int32))

    return lines

#draw cube based on scaled coordinates of cube points
def drawCube(bottom, top,frame,face_color,edge_color):
    thickness=5
    #-1 for fill; 0 for transparent
    
    #Lines connecting top and bottom of cube
    sides= connectCubeCornerstoTag(bottom, top)
    for s in sides: #red faces of cube
        cv2.drawContours(frame,[s],0,face_color,thickness)
        
    #draw square at top of cube and around AR tag (bottom of cube)
    for i in range (4):
        if i==3: #connect last corner to first corner
            cv2.line(frame,tuple(bottom[i]),tuple(bottom[0]),edge_color,thickness)
            cv2.line(frame,tuple(top[i]),tuple(top[0]),edge_color,thickness)
        else:
            cv2.line(frame,tuple(bottom[i]),tuple(bottom[i+1]),edge_color,thickness)
            cv2.line(frame,tuple(top[i]),tuple(top[i+1]),edge_color,thickness)

    return frame

