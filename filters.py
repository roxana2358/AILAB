import dlib
import cv2
import numpy as np

img = cv2.imread("imgs/colori.jpg")

res = np.zeros(img.shape, np.uint8) # creating blank mask for result
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#stored_frame = hsv
# for red
# lower1 = np.array([170,100,20]) # setting lower HSV value
# upper1 = np.array([180,255,255]) # setting upper HSV value
# mask = cv2.inRange(hsv, lower1, upper1) # generating mask

# lower2 = np.array([0,100,20]) # setting lower HSV value
# upper2 = np.array([10,255,255]) # setting upper HSV value
# mask2 = cv2.inRange(hsv, lower2, upper2) # generating mask

# mask = mask + mask2

# for blue
# lower1 = np.array([100,100,20]) # setting lower HSV value
# upper1 = np.array([120,255,255]) # setting upper HSV value
# mask = cv2.inRange(hsv, lower1, upper1) # generating mask

#for green and yellow
lower1 = np.array([20,0,0]) # setting lower HSV value (20,0,20) (46 0 20)
upper1 = np.array([80,255,255]) # setting upper HSV value  (86 255 255)
mask = cv2.inRange(hsv, lower1, upper1) # generating mask

# for yellow
# lower1 = np.array([20,0,100]) # setting lower HSV value
# upper1 = np.array([40,255,255]) # setting upper HSV value
# mask = cv2.inRange(hsv, lower1, upper1) # generating mask

inv_mask = cv2.bitwise_not(mask) # inverting mask
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res1 = cv2.bitwise_and(img, img, mask= mask) # region which has to be in color
#stored_frame = res1
res2 = cv2.bitwise_and(gray, gray, mask= inv_mask) # region which has to be in grayscale
for i in range(3):
    res[:, :, i] = res2 # storing grayscale mask to all three slices
img = cv2.bitwise_or(res1, res) # joining grayscale and color region


# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)              # convert the frame to grayscale
# gray = cv2.medianBlur(gray, 3)                          # apply median filter
# edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)   # detect edges
# color = cv2.bilateralFilter(img, 3, 300, 300)   # apply bilateral filter   
# cartoon = cv2.bitwise_and(color, color, mask=edges)     # combine color image with edges
# cv2.imwrite("imgs/cartoon2.jpg", cartoon)


# cv2.imwrite("imgs/edges.jpg", edges)
# cv2.imwrite("imgs/medianBlur.jpg", gray)
cv2.imshow("img", img)
cv2.waitKey(0)