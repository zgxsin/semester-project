# -------------------------------------------------------------------------------------------------------------------
# Method 2 in OpenCv
# -------------------------------------------------------------------------------------------------------------------
import numpy as np
import cv2 as cv
from PIL import Image
cap = cv.VideoCapture('/Users/zhou/Desktop/data/video_clip/train/IMG_00000.MOV')
fgbg = cv.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
# fgbg = cv.bgsegm.createBackgroundSubtractorGMG(initializationFrames=200, decisionThreshold=0.85)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(4,4))

X = True
count_frame = 0
image_origin_list =[]

cv.namedWindow( 'image', cv.WINDOW_NORMAL )
cv.resizeWindow('image', 1024, 1980)
while(1):
    ret, frame = cap.read()
    # print( "Processing " + str( count_frame ) + "th frame" )
    count_frame = count_frame+1

    fgmask = fgbg.apply(frame)

    if count_frame%30 ==0:
    # print(frame.shape)
        image_origin= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        fgmask = cv.morphologyEx( fgmask, cv.MORPH_OPEN, kernel, iterations=2)
        # fgmask = np.asarray( fgmask == 255, np.uint8 )
        composite = np.concatenate((image_origin, fgmask), axis=0 )


        print(count_frame)
        cv.imshow('image', composite)
        if count_frame == 510:
            composite = Image.fromarray( composite )
            composite.save(str(count_frame) + ".png")
            k = cv.waitKey(100000) & 0xff
        k = cv.waitKey(5) & 0xff
        if k == 27:
            break
cap.release()
cv.destroyAllWindows()


# -------------------------------------------------------------------------------------------------------------------
# Method 3 in openCv
# -------------------------------------------------------------------------------------------------------------------

# import numpy as np
# import cv2 as cv
# import matplotlib
# import matplotlib.pyplot as plt
# # pip3 install Pillow
# import os,sys
# from PIL import Image
# import glob
# # filelist = glob.glob('/Users/zhou/Desktop/untitled folder/')
# # # filelist = 'file1.bmp', 'file2.bmp', 'file3.bmp'
# # filelist = np.sort(filelist).tolist()
# # frame_data = np.array([np.array(Image.open(fname)) for fname in filelist])
#
# # cap = cv.VideoCapture('/Users/zhou/Desktop/untitled folder/image_%05d.png')
# cap = cv.VideoCapture("/Users/zhou/Desktop/data/video_clip/train/IMG_00002.MOV")
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
# fgbg = cv.bgsegm.createBackgroundSubtractorGMG(initializationFrames=200, decisionThreshold=0.85)
# # fgbg = cv.bgsegm.createBackgroundSubtractorGSOC()
#
# # fgbg = cv.bgsegm.createBackgroundSubtractorCNT()
# i = 1
# # attention: dead loop
# while(1):
#     ret, frame = cap.read()
#     if ret:
#         print("Processing "+ str(i)+"th frame")
#         fgmask = fgbg.apply(frame)
#         fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel, iterations=2)
#         # im = Image.fromarray(fgmask)
#         # im.save( "image/image" + str(i) +".png" )
#         cv.imshow('frame',fgmask)
#         i = i + 1
#         k = cv.waitKey(1) & 0xff
#         if k == 27:
#             break
# cap.release()
# cv.destroyAllWindows()



# -------------------------------------------------------------------------------------------------------------------
# Method 1 in openCv
# ----------------------------------------------------------------------------------------------------------------

# import numpy as np
# import cv2 as cv
# cap = cv.VideoCapture('/Users/zhou/Desktop/data/video_clip/train/IMG_00002.MOV')
# fgbg = cv.bgsegm.createBackgroundSubtractorMOG(history=500)
# while(1):
#     ret, frame = cap.read()
#     fgmask = fgbg.apply(frame)
#     cv.imshow('frame',fgmask)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
# cap.release()
# cv.destroyAllWindows()