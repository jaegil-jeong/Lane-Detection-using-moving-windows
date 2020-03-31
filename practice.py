import lanedetection
import cv2
import numpy as np




cap = cv2.VideoCapture('output_l.avi')
frameWidth= 640
frameHeight = 480
# src =np.float32([(0,0.96),(0.41,0.98),(0.23,0.58),(0.57,0.58)])

trackbar_init_value =[(23, 60), (55, 59), (0, 99), (38, 98)]
lanedetection.init_trackbar(trackbar_init_value)
while True :
    ret,img = cap.read()
    # img= cv2.imread('test2.JPG')
    img = cv2.resize(img, (frameWidth, frameHeight), None)
    src =lanedetection.val_trackbar()
    imgWarpPoints = img.copy()
    imgFinal = img.copy()
    imgCanny = img.copy()
    rowWarp = lanedetection.perspective_warp(img, dst_size=(640,480), src = src)
    imgUndis = lanedetection.camera_cali(img)
    imgThreshold = lanedetection.color_filter(imgUndis)

    imgWarp = lanedetection.perspective_warp(imgThreshold, dst_size=(640,480), src=src)
    imgWarpPoints =lanedetection.drawPoints(imgWarpPoints, src)
    boxImg , curve, coeff, ploty = lanedetection.sliding_window(imgWarp, draw_windows=True)


    imgFinal = lanedetection.draw_lanes(img,curve,frameWidth,frameHeight,src= src)
    imgStacked = lanedetection.stackImages(0.5, ([img,imgUndis,imgWarpPoints],
                                             [imgThreshold, imgCanny, rowWarp],
                                             [imgWarp,boxImg,imgFinal]
                                             ))
    print('X=',coeff[0],'*y^2',' + ', coeff[1],'*y', ' + ', coeff[2])
    cv2.imshow('a', imgStacked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()