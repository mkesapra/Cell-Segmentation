import cv2 as cv
import numpy as np

img = cv.imread('images/imagesMQAE.tif')
image_contours = np.zeros((img.shape[1],
                           img.shape[0], 1),
                          np.uint8)

image_binary = np.zeros((img.shape[1],
                         img.shape[0], 1),
                        np.uint8)

for channel in range(img.shape[2]):
    ret, image_thresh = cv.threshold(img[:, :, channel],
                                     38, 255,
                                     cv.THRESH_BINARY)

    contours = cv.findContours(image_thresh, 1, 1)[0]   
    cv.drawContours(image_contours,
                    contours, -1,
                    (255,255,255), 3)

contours = cv.findContours(image_contours, cv.RETR_LIST,
                           cv.CHAIN_APPROX_SIMPLE)[0]

cv.drawContours(image_binary, [max(contours, key = cv.contourArea)],
                -1, (255, 255, 255), -1)

#cv.imwrite('LPR.jpg', image_binary)
cv.imshow('LPR', image_binary)
cv.waitKey(0) & 0xFF is 27
cv.destroyAllWindows()