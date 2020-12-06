import cv2 as cv

video = cv.VideoCapture("highway.mp4")
substractor = cv.createBackgroundSubtractorMOG2(history=20,varThreshold=20,detectShadows=True)

while True:
    ret,frame = video.read()
    cv.imshow("FRAME",frame)
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    gray_frame = cv.blur(gray_frame,(5,5))

    mask = substractor.apply(gray_frame)

    contours,hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame,contours,-1,(255,0,0),2)

    cv.imshow("BS",mask)
    cv.imshow("CARS FOUND",frame)

    if cv.waitKey(5) == 27:
        break

video.release()
cv.destroyAllWindows()