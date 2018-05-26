import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from os import listdir
from os.path import isfile, join
# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
# construct the argument parse and parse the arguments

# load the image
plt.figure()
image = cv2.imread("/Users/zhou/Desktop/Mask_RCNN_Edit/dataset/train_mask/000050.png")
# plt.imshow(image, 'gray')
# plt.show()
image = image > 0
image = np.asarray(image, np.uint8)

# mask_temp = skimage.io.imread("/Users/zhou/Desktop/Mask_RCNN_Upload/dataset/train_mask/000050.png", as_grey=True)
# find all the 'black' shapes in the image
lower = 0.3
upper = 1
shapeMask = cv2.inRange(image, lower, upper)
# print(shapeMask)

# find the contours in the mask
_, cnts, _ = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE )
print ("I found %d black shapes" % (len( cnts )))
# cv2.imshow( "Mask", shapeMask )

# loop over the contours
for c in cnts:
    # draw the contour and show it
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2 )
    cv2.imshow("Image", image)
    cv2.waitKey(1000)
# cv2.waitKey(0)
#### https://www.pyimagesearch.com/2014/10/20/finding-shapes-images-using-python-opencv/