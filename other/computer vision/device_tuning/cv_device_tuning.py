import sys
print(sys.version)
import cv2
print('OpenCV Version: {}'.format(cv2.__version__))
import numpy as np
from matplotlib import pyplot as plt

def cvt_to_uint8(img):
    pass
    new_img = img
    new_img /= np.max(abs(new_img))
    new_img  = np.array((new_img+1)/2*255, dtype=np.uint8)
    return new_img


img = cv2.imread('sweep.png')
input_img = img
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#
# Create a mask using a laplacian and further filtering
#

laplacian = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=7)
laplacian = cv2.Sobel(laplacian, cv2.CV_64F, 0, 3, ksize = 7)
laplacian = cvt_to_uint8(laplacian)
_, laplacian = cv2.threshold(laplacian, np.max(laplacian)*0.60, 255, cv2.THRESH_BINARY)
laplacian_overlay = laplacian.copy()
cv2.imshow('High dY', laplacian)
cv2.waitKey()
cv2.imwrite('laplacian_overlay.png', laplacian_overlay)

#
# Create a mask using a single derivative and further filtering
#

# Calculate the image's first derivative along y
dy1 = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize = 7)
dy1 = cvt_to_uint8(dy1)
dy1 = cv2.medianBlur(dy1, 3)
cv2.imwrite('dy1_8bit.png', dy1)

# Threshold the filtered image
#threshold_y = cv2.adaptiveThreshold(dy1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
#                                    cv2.THRESH_BINARY_INV, 7, 1)
_, threshold_y = cv2.threshold(dy1, np.max(dy1)*3/4, 255, cv2.THRESH_BINARY)

cv2.imshow('threshold_y', threshold_y)
cv2.waitKey()

output_laplacian = input_img.copy()
output_sobel = input_img.copy()

output_laplacian[laplacian_overlay != 0] = (0, 255, 0)
cv2.imshow('Overlayed Image (Laplacian)', output_laplacian)

output_sobel[threshold_y != 0] = (0,255,0)
cv2.imshow('Overlayed Image (Sobel (Y1))', output_sobel)
cv2.waitKey()

cv2.imwrite('output_laplacian.png', output_laplacian)
cv2.imwrite('output_sobel.png', output_sobel)


#
# Try to see what feature detection can liberate from our image
#

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(dy1, None)
kp.sort(key=lambda x:x.response, reverse=True)
img = cv2.drawKeypoints(gray_img,kp[0:100], img)


# Try and find keypoints across entire image

dim = 5 # The grid dimension
n_regions = dim**2
width, height = np.shape(gray_img)
all_kps = []
kp_test_img = gray_img
draw_region = False

for r in range(n_regions):
    # Calculate corners of region
    x_0 = (r % dim) * width/dim
    y_0 = (r / dim) * height/dim
    x_1 = ((r % dim) + 1) * width/dim
    y_1 = ((r / dim) + 1) * height/dim
    #print('({},{}) -> ({},{})'.format(x_0,y_0, x_1,y_1))

    # Define mask for region
    mask = np.zeros(kp_test_img.shape[:2], np.uint8)
    mask[x_0:x_1,y_0:y_1] = 255
    roi = cv2.bitwise_and(kp_test_img, mask)

    # Find keypoints for ROI
    roi_img = kp_test_img[x_0:x_1,y_0:y_1]
    _, roi_img = cv2.threshold(roi_img, 30, 255, cv2.THRESH_BINARY)
    kp = sift.detect(roi_img, None)

    # Sort keypoints by how 'important' they are
    # response is an attribute of keypoints which is larger for 
    # 'more interesting' keypoints
    kp.sort(key=lambda x: x.response, reverse=True)

    for k in kp:
        k.pt = (k.pt[0]+y_0, k.pt[1]+x_0)

    # Draw keypoints
    roi_img = cv2.drawKeypoints(kp_test_img,kp, img)
    if draw_region:
        cv2.imshow('Region of Interest', roi)
        cv2.imshow('Keypoints', roi_img)
        cv2.waitKey()

    kp = kp[:30]
    all_kps += kp

# Draw all keypoints found for image
kp = all_kps
cv2.drawKeypoints(gray_img,kp, img)
cv2.imwrite('sift_keypoints.jpg',img)
cv2.imshow('Keypoints', img)
cv2.waitKey()

## Exit early
exit()

#
# Trying a Hough Line detector
# (this doesn't work very well
#

_, mask = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
test_img = cv2.bitwise_and(gray_img, gray_img, mask=mask)
edges = cv2.Canny(test_img, 20, 150, apertureSize=3)
plt.imshow(edges)
plt.show()

lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength = 15, maxLineGap = 10)
output_img = input_img.copy()

def draw_line(*args):
    if (len(args) == 2):
        rho = args[0]
        theta = args[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

    else:
        x1 = args[0]
        y1 = args[1]
        x2 = args[2]
        y2 = args[3]

    cv2.line(output_img,(x1,y1),(x2,y2),(0,255,0),2)

if lines is None:
    print('No lines found in image.')
else:
    for line in lines:
        draw_line(*line[0])

    cv2.imshow('Output', output_img)

cv2.waitKey()
