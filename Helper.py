import cv2 as cv
import numpy as np
def Preprocess(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(frame, (7,7), 0)
    th= cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    th = cv.morphologyEx(th, cv.MORPH_OPEN, kernel,8)
    return th

def Order_wise(a) :
    if a is not None:
        a = a.reshape((4, 2))
        ar = np.zeros(a.shape)
        ar[0] = a[np.sum(a, 1).argmin()]
        ar[3] = a[np.sum(a, 1).argmax()]               # ar[1] = a[np.diff(a,0).argmin()] # ar[2] = a[np.diff(a,0).argmax()]
        x = np.sum(a, 1).argmin()
        y=  np.sum(a, 1).argmax()
        a1 = [x,y]
        a2 =[0,1,2,3]
        id = [i for i in a2 if i not in a1]
        a2 = a[np.array(id),0:2]
        ar[1] = a2[a2[:][0].argmax()]
        ar[2] = a2[a2[:][0].argmin()]
    else:
        return None
    return ar.astype(np.float32)

def Segmentation(processed,frame):

    contours, hier = cv.findContours(processed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    frame2 = frame.copy()
    corners = None
    segmented=None
    imgs =[]
    cnt= None
    max_area=25000
    for idx,i in enumerate(contours):
        area = cv.contourArea(i)
        if area > 30000 and area < (processed.shape[0] * processed.shape[1] - 25000):
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.13 * peri, True)
            if area>max_area and len(approx) == 4:
                cnt = i
                max_area = area
                corners = approx
    if cnt is not None and corners is not None:
        cv.drawContours(frame, cnt, -1, (0, 255, 0), 4)
        mat = cv.getPerspectiveTransform(Order_wise(corners), np.float32([[0, 0], [594, 0], [0, 594], [594, 594]]))
        segmented = cv.warpPerspective(frame2, mat, (594, 594))
        segmented_th = Preprocess(segmented)
        segmented_th = cv.bitwise_not(segmented_th)

        imgs = Grid_split(segmented_th)
    return Order_wise(corners),segmented,imgs

def Grid_split(sudoku):
    imgs =[]
    sudoku = np.vsplit(sudoku, 9)
    for i in sudoku :
        imgs.append(np.hsplit(i, 9))
    return np.array(imgs).reshape(-1,66,66)









# def Segmentation(processed,frame):
#
#     contours, hier = cv.findContours(processed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     frame2 = frame.copy()
#     corners = None
#     segmented=None
#     imgs =[]
#     cnt= None
#     max_area=25000
#     for idx,i in enumerate(contours):
#         area = cv.contourArea(i)
#         if area > 30000 :#and area < (processed.shape[0] * processed.shape[1] - 45000):
#             peri = cv.arcLength(i, True)
#             approx = cv.approxPolyDP(i, 0.1 * peri, True)
#             if area>max_area and len(approx) == 4:
#                 cnt = i
#                 max_area = area
#                 corners = approx
#     if cnt is not None:
#         cv.drawContours(frame, cnt, -1, (0, 255, 0), 4)
#         mat = cv.getPerspectiveTransform(Order_wise(corners), np.float32([[0, 0], [594, 0], [0, 594], [594, 594]]))
#         segmented = cv.warpPerspective(frame2, mat, (594, 594))
#         segmented_th = Preprocess(segmented)
#         segmented_th = cv.bitwise_not(segmented_th)
#         imgs = Grid_split(segmented_th)
#     return Order_wise(corners),segmented,imgs
#


