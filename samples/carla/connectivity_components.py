
# Import the cv2 library
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the image you want connected components of
src= cv2.imread("/Users/zhou/Desktop/Mask_RCNN_Upload/dataset/train_mask/000050.png", 0)
# Threshold it so it becomes binary
# ret, thresh = cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# You need to choose 4 or 8 for connectivity type
src = src > 0
src = np.asarray(src, np.uint8)
connectivity = 8
# Perform the operation
output = cv2.connectedComponents(src, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
labels = output[1]
plt.figure()
plt.imshow(src)
count = 0
for i in range(num_labels):
    # robust to noise
    if np.sum(labels == i) >=10:
        plt.imshow(labels==i)
        count = count + 1
plt.show()

