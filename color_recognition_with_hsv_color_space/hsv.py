import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0)

print("Press any key to exit")

min_color = np.array([110,150,50])
max_color = np.array([130,255,255])

ret,frame = camera.read()
while ret and cv.waitKey(1) == -1 :
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv,min_color,max_color)
    last = cv.bitwise_and(frame,frame,mask=mask)
    cv.imshow("LAST LIVE",last)
    cv.imshow("MASK LIVE",mask)
    cv.imshow("HSV LIVE",hsv)
    cv.imshow("MyWindow",frame)
    ret,frame = camera.read()

cv.destroyAllWindows()
