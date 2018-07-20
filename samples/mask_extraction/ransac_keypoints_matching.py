import numpy as np
import cv2
from matplotlib import pyplot as plt


# -------------------------------------------------------------------------------------------------------------------
# display oringinal image (part1)
# -------------------------------------------------------------------------------------------------------------------
# img1 = cv2.imread('opencv/samples/data/box.png', 0)
# cv2.imshow('1', img1)
# cv2.waitKey(0)

img1 = cv2.imread('/Users/zhou/Desktop/testset/dynamic/000040.png' , cv2.IMREAD_GRAYSCALE)
print('image 1 shape : ', img1.shape)
# img2 = cv2.imread('opencv/samples/data/box_in_scene.png', 0)
img2 = cv2.imread('/Users/zhou/Desktop/testset/dynamic/000180.png', 0)

img3 = cv2.imread("/Users/zhou/Desktop/Pictures/IMG_00000.jpg", 0)

# apply image enhancement
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
img1 = clahe.apply(img1)
img2 = clahe.apply(img2)
img1_origin = img1.copy()
img2_origin = img2.copy()
print('image 2 shape : ', img2.shape)

fig0 = plt.figure("original images")
ax1 = fig0.add_subplot(1,3,1)
ax1.set_title("image1")
ax1.imshow(img1, 'gray')
ax2 = fig0.add_subplot(1,3,2)
ax2.set_title("image2")
ax2.imshow(img2, 'gray')


# -------------------------------------------------------------------------------------------------------------------
# detect sift features and draw keypoints (part2)
# -------------------------------------------------------------------------------------------------------------------
ref_sizey, ref_sizex = (max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1]))
sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=10)
# reference: https://docs.opencv.org/master/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
# find the kepoints and descriptors with SIFT
kp1, des1 =sift.detectAndCompute(img1, None)
kp2, des2 =sift.detectAndCompute(img2, None)

kp3, des3 =sift.detectAndCompute(img3, None)
print("The number of Sift Features of image1: ", len(kp1))
print("The number of Sift Features of image2: ", len(kp2))
img1_ = cv2.drawKeypoints(img1, kp1, None, color=(255,0,0))
# show the image with keypoints marked
# cv2.imshow('keypoints of img1 usig sift', img1_)
fig1 =plt.figure("keypoints in images")
# fig = plt.figure(200)
ax_1 = fig1.add_subplot(1,2,1)
ax_1.set_title('keypoints of img1 usig sift')
ax_1.imshow(img1_, 'gray')

# cv2.waitKey(0)
img2_ = cv2.drawKeypoints(img2, kp2, None, color=(255,0,0))
# cv2.imshow("keypoints of img2 using sift", img2_)
# stitcher = cv2.createStitcher(False)
# (_result, pano) = stitcher.stitch((img1, img2))
# cv2.imshow("keypoints of two images using sift", pano)
# cv2.waitKey(0)
# print(kp1)

ax_2 = fig1.add_subplot(1,2,2)
ax_2.set_title('keypoints of img2 usig sift')
ax_2.imshow(img2_, 'gray')


img3_ = cv2.drawKeypoints(img3, kp3, None, color=(255,0,0))
# show the image with keypoints marked
# cv2.imshow('keypoints of img1 usig sift', img1_)
fig1 =plt.figure("presentation_image")
plt.title("Original image with SIFT features")
plt.imshow(img3_, 'gray')

# -------------------------------------------------------------------------------------------------------------------
# find matches and apply ratio test (part3)
# -------------------------------------------------------------------------------------------------------------------
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
total_match = []
for m,n in matches:
    # cancel ratio test
    total_match.append(m)
    if m.distance < 0.75*n.distance:
        good.append(m)

print("The number of matching pairs after ratio testing: ", len(good))
fig2 =plt.figure("keypoints matching")
# plt.figure(300)
axs_1 = fig2.add_subplot(2,1,1)

# -------------------------------------------------------------------------------------------------------------------
# apply ransac and draw (part4)
# -------------------------------------------------------------------------------------------------------------------
MIN_MATCH_COUNT = 30
if len(good) > MIN_MATCH_COUNT:
    # show the images matching before ransac
    # matchesMask_all = np.ones((len(good),), dtype = int).tolist()
    # matchesMask_all[20]=1
    img4 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, matchColor = (255, 0 ,0), singlePointColor = (0, 255 ,0))
    axs_1.set_title('image matching before ransac, after ratio testing')
    axs_1.imshow(img4, 'gray')
    # plt.show()
    # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # M:transformation matrix from image1 to image2
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 15, maxIters=10000)
    matchesMask = np.asarray(mask.ravel().tolist())
    matching_points = len(matchesMask[np.nonzero(matchesMask)])
    print("The number of matching pairs after RANSAC: ", matching_points)
    # select several matches to draw
    '''
    idxs = np.where(matchesMask==1)[0]
    if len(idxs) > 10:
        idxs_short = idxs[:30]
        matchesMask[:] = 0
        matchesMask[idxs_short] = 1
    '''
    matchesMask = matchesMask.tolist()
    # get the after_tranformation image
    img1_trans = cv2.warpPerspective(img1, M, (800, 600))
    ax3 = fig0.add_subplot(1, 3, 3)
    ax3.set_title("transforming image1")
    ax3.imshow(img1_trans, 'gray')

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    dst_array = np.int32(dst)
    # compute the size for computing the mask contours
    size_x = max
    # print(np.int32(dst))
    img2_line = cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 4, cv2.LINE_AA)


else:
    print('Not enough matches are found - {}/{}'.format(len(good), MIN_MATCH_COUNT))
    matchesMask = None



draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255, 0 ,0),
                   matchesMask = matchesMask,
                   flags = 2)
img3 = cv2.drawMatches(img1, kp1, img2_line, kp2, good, None, **draw_params)
axs_2 = fig2.add_subplot(2,1,2)
axs_2.set_title('image matching after ransac')
axs_2.imshow(img3, 'gray')

# draw after_ransac keypoints matching in among the images which are drawn with original sift keypoints
index_pts = np.asarray(matchesMask)
img1_points = []
img2_points = []
# define the matching keypoints set after ratio test and before ransac
total_matchPts1 = []
total_matchPts2 = []
for i in range(len(index_pts)):
    if index_pts[i] == 1:
        # find the matching keypoints after ransac for each image
        img1_points.append(kp1[good[i].queryIdx])
        img2_points.append(kp2[good[i].trainIdx])
    else:
        total_matchPts1.append( kp1[good[i].queryIdx] )
        total_matchPts2.append( kp2[good[i].trainIdx] )

# draw all the matching points except those matching points after ransac in the image1 and image2
img1_1_ = cv2.drawKeypoints(img1, total_matchPts1, None, color=(255,0,0))
img2_2_ = cv2.drawKeypoints(img2_origin, total_matchPts2, None, color=(255,0,0))

img1_final= cv2.drawKeypoints(img1_1_, img1_points, None, color=(0,255,0))
img1_final_trans = cv2.warpPerspective(img1_final, M, (800, 600))




# -------------------------------------------------------------------------------------------------------------------
# draw difference image
# -------------------------------------------------------------------------------------------------------------------
img1_warp = cv2.warpPerspective(img1_origin, M, (800, 600))
img1_warp = img1_warp/img1_warp.max()
img2_origin_nml = img2_origin/img2_origin.max()
diff_image = abs(img1_warp -img2_origin_nml)
# for i in range(diff_image.shape[0]):
#     for j in range(diff_image.shape[1]):
#         if diff_image[i,j] >= 0.4:
#             diff_image[i,j] = 1
#         else:
#             diff_image[i, j] = 0
plt.figure("diffenrence image")
plt.imshow(diff_image, 'gray')

dy_keypoints_img1 = np.asarray([total_matchPts1[i].pt for i in range(len(total_matchPts1))], dtype=np.float32).reshape(-1, 1, 2)
# after perspective transformation, keypoints belong to dynamic objects: transDy_keypoints_img1
transDy_keypoints_img1 = cv2.perspectiveTransform(dy_keypoints_img1, M)
#  transDy_keypoints_img1 -> tuple
tp_transDy_keypoints_img1 = tuple(map(tuple, transDy_keypoints_img1.reshape(-1, 2)))
keypoints_img1 = np.asarray([img1_points[i].pt for i in range(len(img1_points))], dtype=np.float32).reshape(-1,1, 2)
# after perspective transformation. keypoints belong to stable objects: transKeypoints_img1
transKeypoints_img1 = cv2.perspectiveTransform(keypoints_img1, M)
# transKeypoints_img1 -> tuple
tp_transKeypoints_img1 = tuple(map(tuple, transKeypoints_img1.reshape(-1, 2)))


img2_final= cv2.drawKeypoints(img2_2_, img2_points, None, color=(0,255,0))

# keypoints belong to dynamic objects: dy_keypoints_img2
dy_keypoints_img2 = np.asarray([total_matchPts2[i].pt for i in range(len(total_matchPts2))], dtype=np.float32).reshape(-1, 2)
# dy_keypoints_img2 -> tuple type
tp_dy_keypoints_img2 = tuple(map(tuple, dy_keypoints_img2))
# keypoints belong to stable objects: keypoints_img2
keypoints_img2 = np.asarray([img2_points[i].pt for i in range(len(img2_points))], dtype=np.float32).reshape(-1, 2)
# keypoints_img2 -> tuple type
tp_keypoints_img2 = tuple(map(tuple, keypoints_img2))


plt.figure("final_image1")
plt.imshow(img1_final)
plt.figure("final_image2")
plt.imshow(img2_final)
plt.figure("final_image1_trans")
plt.imshow(img1_final_trans)

# find similar region
# cv2.pointPolygonTest()
# define a contour genrate function
# define mask size
# width = 5
# height = 5
# contour = []
# def ContourMask(size = 5):
#     for i in range(ref_sizex-size):
#         for j in range(ref_sizey-size):
#             contour.append(np.array([[i,j],[i+size,j],[i+size, j +size],[i, j+size]], dtype=np.int32))
#
#     return contour
#
#
# contour = ContourMask(size = 5)
# L = len(contour)
#
# img2_pitchScore = np.zeros(shape=(L,), dtype=np.float32)
# img1_pitchScore = np.zeros(shape=(L,), dtype=np.float32)
# for i in range(len(contour)):
#     k_img2_x = 0
#     k_img2_y = 0
#     for j in range(len(tp_keypoints_img2)):
#         symbol1 = cv2.pointPolygonTest(contour[i], tp_keypoints_img2[j], measureDist = False)
#         if symbol1:
#             k_img2_x = k_img2_x + 1
#     for q in range(len(tp_dy_keypoints_img2)):
#         symbol2 = cv2.pointPolygonTest(contour[i], tp_dy_keypoints_img2[q], measureDist=False)
#         if symbol2:
#             k_img2_y = k_img2_y + 1
#     img2_pitchScore[i] = k_img2_x
#
#
# for i in range(len(contour)):
#     k_img1_x = 0
#     k_img1_y = 0
#     for j in range(len(tp_transKeypoints_img1)):
#         symbol1 = cv2.pointPolygonTest(contour[i], tp_transKeypoints_img1[j], measureDist = False)
#         if symbol1:
#             k_img1_x = k_img1_x + 1
#     for q in range(len(tp_transDy_keypoints_img1)):
#         symbol2 = cv2.pointPolygonTest(contour[i], tp_transDy_keypoints_img1[q], measureDist=False)
#         if symbol2:
#             k_img1_y = k_img1_y + 1
#     img1_pitchScore[i] = k_img1_x

plt.show()


