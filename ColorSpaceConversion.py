import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    ret,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([125,43,46])
    upper_blue = np.array([155,255,255])
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    res = cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow('test1',frame)
    cv2.imshow('test2',mask)
    cv2.imshow('test3',res)
    k = cv2.waitKey(5)&0xFF
    if k == 27:
        break
cv2.destroyAllWindows()