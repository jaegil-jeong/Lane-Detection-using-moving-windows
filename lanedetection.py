import cv2
import numpy as np
import pickle

def nothing(x):
    pass

def camera_cali(img, cal_dir = 'cal_pickle.p'):
    # Camera Calibration for test sample
    # the number of inside corner in x, y direction
    with open(cal_dir, mode = 'rb') as f:
        file= pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


def color_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerYellow = np.array([0, 70, 100])
    upperYellow = np.array([179, 255, 200])
    lowerWhite = np.array([0, 0, 200])
    upperWhite = np.array([255, 255, 255])
    maskedWhite = cv2.inRange(hsv, lowerWhite, upperWhite)
    maskedYellow = cv2.inRange(hsv, lowerYellow, upperYellow)
    combinedImage = cv2.bitwise_or(maskedWhite, maskedYellow)
    return maskedYellow

def threshodling(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5,5))
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),0)
    imgCanny = cv2.Canny(imgBlur,50 , 100)
    colorFiltered = color_filter(img)
    combinedImage = cv2.bitwise_or(imgCanny,colorFiltered)
    return combinedImage

def drawPoints(img, src):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    # src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    src = src * img_size
    for x in range(0, 4):
        cv2.circle(img, (int(src[x][0]), int(src[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img
def perspective_warp(img,
                     dst_size=(1280,720),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst= np.float32([(0,0), (1,0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = img_size*src
    dst = dst* np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src,dst)
    warped =cv2.warpPerspective(img, M, dst_size,flags=cv2.INTER_LINEAR)

    return warped

def inv_perspective_warp(img,
                         dst_size=(1280,720),
                         src=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
                         dst=np.float32([(0.2, 0.65), (0.65, 0.65), (0.1, 1), (1, 1)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src*img_size
    dst = dst* np.float32(dst_size)
    inv_M = cv2.getPerspectiveTransform(dst, src)
    inv_wapred = cv2.warpPerspective(img, inv_M, dst_size,flags=cv2.INTER_LINEAR)
    return inv_wapred

a, b, c = [], [], []

window_width =98
prev_lane_inds = []
def sliding_window(img ,windows_num=20, margin =int(window_width/2) , minpix =300, draw_windows = True):
    global a, b, c
    window_height = np.int(img.shape[0] / windows_num)
    fit = np.empty(3)
    out_img = np.dstack((img,img,img))* 255
    histogram = np.sum(img[int(3*img.shape[0]/4):,:int(img.shape[1]/2+100)], axis = 0)

    base_x =np.argmax(histogram)
    window_height = np.int(img.shape[0]/windows_num)

    nonzero = np.nonzero(img)
    nonzeroy = np.array(nonzero[0]) # y index of nonzero pixel
    nonzerox = np.array(nonzero[1]) # x index of nonzero pixel

    current_x = base_x

    lane_inds =[]

    for window in range(windows_num):
        # Identify winodw boundaries in x and y
        win_lower_y = img.shape[0] - (window +1)*window_height
        win_upper_y = img.shape[0] -window*window_height
        win_lower_x = int(current_x -margin)
        win_upper_x = int(current_x + margin)
        # img_layer = np.sum(img[int(img.shape[0] - (window+1)*window_height):
        #                    int(img.shape[0]-window*window_height),:int(img.shape[1]/2+100)],axis = 0)
        # current_x = np.argmax(img_layer)
        # offset = window_width/2
        # win_lower_x = int(max(current_x+offset-margin,0))
        # win_upper_x =int(min(current_x +offset+margin,img.shape[1]))
        # win_lower_y = img.shape[0] - (window +1)*window_height
        # win_upper_y = img.shape[0] -window*window_height
        #Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img, (win_lower_x, win_lower_y), (win_upper_x, win_upper_y),
                          (100,255,255),1)

            # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >=win_lower_y) & (nonzeroy < win_upper_y) & (nonzerox >= win_lower_x) & (nonzerox < win_upper_x)).nonzero()[0]

        lane_inds.append(good_inds)
        cnt =0
        # If you fond > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            current_x = np.int(np.mean(nonzerox[good_inds]))
            cnt = cnt+1

    if len(lane_inds)>0:
        prev_lane_inds = lane_inds.copy()
    else : lane_inds = prev_lane_inds.copy()


    lane_inds = np.concatenate(lane_inds) # nonzero pixel index within a window box  all  concatenate together

    # Extract left and right line pixel
    x_lane_pos = nonzerox[lane_inds]
    y_lane_pos = nonzeroy[lane_inds]

    if x_lane_pos.size :
        fit_coeff= np.polyfit(y_lane_pos, x_lane_pos, 2)
        a.append(fit_coeff[0])
        b.append(fit_coeff[1])
        c.append(fit_coeff[2])

        # mean coefficient of lane function of recent 10 frame
        fit[0]= np.mean(a[-10:])
        fit[1] = np.mean(b[-10:])
        fit[2] = np.mean(c[-10:])

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0] -1, img.shape[0])

        fit_x = fit[0]*ploty**2 + fit[1]*ploty + fit[2]  # x = ay^2 + by + c

        out_img[nonzeroy[lane_inds],nonzerox[lane_inds]] = [255, 0, 0]

        return out_img, fit_x, fit, ploty
    else:
        return img, 0,0,0

def get_curve(img, fit_x):
    ploty = np.linspace(0,img.shape[0] -1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix =1/ img.shape[0] # meters per pixel
    xm_per_pix = 0.1/img.shape[0]

    # Fit new polymomials to x, y in world space

    fit_car = np.polyfit(ploty * ym_per_pix, fit_x*xm_per_pix, 2)



    fit_x_int = fit_car[0]*img.shape[0]**2 + fit_car[1]*img.shape[0] + fit_car[2]
    return fit_x_int

def draw_lanes(img, fit_x, frameWidth, frameHeight, src):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)

    # left = np.array([np.transpose([fit_x, ploty])])
    #
    # right = np.array([np.transpose(np.vstack([fit_x+5, ploty]))])
    # points = np.hstack((left, right))
    #
    # cv2.fillPoly(color_img, np.int_(points), (0, 200, 0))

    fit_x = np.array(fit_x,np.int32)
    lane = np.array(list(zip(np.concatenate((fit_x- window_width/2, fit_x[::-1]+ window_width/2),axis=0),
                             np.concatenate((ploty,ploty[::-1]),axis=0))), np.int32)
    inv_perspective = inv_perspective_warp(color_img, (frameWidth, frameHeight), dst=src)
    road = np.zeros_like(img)
    cv2.fillPoly(road, [lane], color=[0, 228, 225])
    birdLane = road.copy()
    dst =np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
    img_size = np.float32([(img.shape[1],img.shape[0])])
    dst_size = (640, 480)
    dst = dst * np.float32(dst_size)
    src = src * img_size
    inv_M = cv2.getPerspectiveTransform(dst,src)

    inv_perspective = cv2.warpPerspective(road,inv_M,dst_size,flags=cv2.INTER_LINEAR)
    base = cv2.addWeighted(img, 1.0, inv_perspective, -1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, inv_perspective, 1.0, 0.0)
    return result


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def init_trackbar(init_values):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars",360, 360)
    cv2.createTrackbar("top1_x","Trackbars",init_values[0][0],100,nothing)
    cv2.createTrackbar("top1_y", "Trackbars", init_values[0][1], 100, nothing)
    cv2.createTrackbar("top2_x", "Trackbars", init_values[1][0], 100, nothing)
    cv2.createTrackbar("top2_y", "Trackbars", init_values[1][1], 100, nothing)
    cv2.createTrackbar("bottom1_x", "Trackbars", init_values[2][0], 100, nothing)
    cv2.createTrackbar("bottom1_y", "Trackbars", init_values[2][1], 100, nothing)
    cv2.createTrackbar("bottom2_x", "Trackbars", init_values[3][0], 100, nothing)
    cv2.createTrackbar("bottom2_y", "Trackbars", init_values[3][1], 100, nothing)

def val_trackbar():
    top1_x = cv2.getTrackbarPos('top1_x','Trackbars')
    top1_y = cv2.getTrackbarPos('top1_y', 'Trackbars')
    top2_x = cv2.getTrackbarPos('top2_x', 'Trackbars')
    top2_y = cv2.getTrackbarPos('top2_y', 'Trackbars')
    bottom1_x = cv2.getTrackbarPos('bottom1_x', 'Trackbars')
    bottom1_y = cv2.getTrackbarPos('bottom1_y', 'Trackbars')
    bottom2_x = cv2.getTrackbarPos('bottom2_x', 'Trackbars')
    bottom2_y = cv2.getTrackbarPos('bottom2_y', 'Trackbars')

    src = np.float32([(top1_x/100,top1_y/100), (top2_x/100,top2_y/100),
                      (bottom1_x/100,bottom1_y/100), (bottom2_x/100, bottom2_y/100)])
    return src