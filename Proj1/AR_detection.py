import numpy as np
import cv2
import copy
import utils
import matplotlib.pyplot as plt

#intrinsic matrix
K = np.array([[1346.100595,0,932.1633975],
    [0,1355.933136,654.8986796],
    [0,0,1]])
#check what output video
#change False if there is no need to create videos
show_contours = True
show_Testudo = True
show_cube = True
start = 1
#input image
testudo=cv2.imread('testudo.png')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#input video
vidcap = cv2.VideoCapture('1tagvideo.mp4')
#generate videos
if show_contours or show_Testudo or show_cube:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps_out = 29
    print("Take some time to create video")
if show_contours:
    videoname = ('proj1_contours')
    output = cv2.VideoWriter(str(videoname)+".avi", fourcc, fps_out, (1920,1080))
if show_Testudo:
    videoname1 = ('proj1_testudo')
    out1 = cv2.VideoWriter(str(videoname1)+".avi", fourcc, fps_out, (1920,1080))
if show_cube:
    videoname2 = ('proj1_cube')
    out2 = cv2.VideoWriter(str(videoname2)+".avi", fourcc, fps_out, (1920,1080))

#start the frame of video
vidcap.set(1,start)
count = start
#the video error
if (vidcap.isOpened() == False):
    print("e")

while(vidcap.isOpened()):
    count+=1
    success, image = vidcap.read()
    if success:
        #output image of frame 3
        if count == 3:
            cv2.imwrite("frame.jpg", image)
        #get height and weight of image
        hh, ww = image.shape[:2]
        #threshold the image and do AR detection with fft
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        img_back, contours, magnitude = utils.AR_Code_detection(thresh)
        #store image output of fft edge detection
        if show_contours:
            img_plus_contours = img_back.copy()
        #find April tag corners
        corners = utils.find_tag(contours)
        corners = np.array(corners)
        num_point = utils.find_points(image, corners)
        dim = int(np.sqrt(num_point))
        #do homography
        H = utils.solveHomography(corners, dim)
        H_inv = np.linalg.inv(H)
        #warp the image
        square_img = utils.warp(H_inv, image, dim, dim)
        tag_img, tag_id_binary, orientation = utils.Decode_custom_AR_tag(square_img)
        #input testudo into April tag
        if show_Testudo:
            img_match_orientation=utils.rotateTestudo(testudo, orientation)
            h=image.shape[0]
            w=image.shape[1]
            testudo_dim=img_match_orientation.shape[0]
            H_testudo=utils.solveHomography(corners,testudo_dim)
            warped_frame = utils.warp(H_testudo, img_match_orientation,h,w)
            img_plus_testudo = image.copy()
            blank_frame=cv2.drawContours(img_plus_testudo,[corners], -1,(0),thickness=-1)
            img_plus_testudo=cv2.bitwise_or(warped_frame,blank_frame)
            out1.write(img_plus_testudo)
        #input cube into April tag
        if show_cube:
            edge_color=(0,0,255)
            face_color=(0,0,255)
            cube_dim=dim
            cube_height=np.array([-(cube_dim-1),-(cube_dim-1),-(cube_dim-1),-(cube_dim-1)]).reshape(-1,1)
            cube_corners=np.concatenate((corners[0], cube_height), axis=1)

            H_cube = utils.solveHomographyCube(corners[0],cube_corners)
            P=utils.solveProjectionMatrix(K, H_cube)
            cube_corners_P=utils.projectionPoints(cube_corners,P)
            img_plus_cube=utils.drawCube(corners[0],cube_corners_P, image,face_color, edge_color)
            out2.write(img_plus_cube)
        #save images during the process of frame 3 
        if count==3:
            cv2.imwrite('thresh.jpg', thresh)
            cv2.imwrite('AR_tag_img_frame.jpg', square_img)
            cv2.imwrite('AR_tag_description.jpg', tag_img)
            if show_contours:
                cv2.imwrite('contours_img_frame.jpg', img_plus_contours)
            if show_Testudo:
                cv2.imwrite('testudo_img_frame.jpg', img_plus_testudo)
            if show_cube:
                cv2.imwrite('cube_img_frame.jpg', img_plus_cube)
        if show_contours:
            output.write(np.uint8(img_plus_contours))
    else:
        vidcap.release()

    if cv2.waitKey(1) == ord('q'):
        vidcap.release()
        if show_contours:
            output.release()
        if show_Testudo:
            out1.release()
        if show_cube:
            out2.release()
#print("H", H)
#plt.subplot(131), plt.imshow(image, cmap = 'gray')
#plt.title('Input'), plt.xticks([]), plt.yticks([])
#plt.subplot(132), plt.imshow(magnitude, cmap = 'gray')
#plt.title('magnitude spectrum'), plt.xticks([]), plt.yticks([])
#plt.subplot(133), plt.imshow(img_back, cmap = 'gray')
#plt.title('edge and corner detection'), plt.xticks([]), plt.yticks([])
#plt.show()
#vidcap.release()
if show_contours:
    output.release()
if show_Testudo:
    out1.release()
if show_cube:
    out2.release()
cv2.destroyAllWindows()
