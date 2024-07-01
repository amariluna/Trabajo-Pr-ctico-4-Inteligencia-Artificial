import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('circles.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_blurred = cv2.medianBlur(gray, 5)

detected_circles = cv2.HoughCircles(gray_blurred, 
                                    cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                    param1=50, param2=30, minRadius=1, maxRadius=40)

if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        cv2.circle(image, (a, b), r, (0, 255, 0), 2)
        cv2.circle(image, (a, b), 1, (0, 0, 255), 3)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
