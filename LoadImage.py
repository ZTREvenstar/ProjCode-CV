from cv2 import cv2
import numpy as np

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(res, (x, y), 100, (255, 0, 0), -1)
        
        
img = cv2.imread('Water-pause-times.png', -1)
res = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
cv2.namedWindow('pic1')
cv2.setMouseCallback('pic1', draw_circle)

while(1):
    cv2.imshow('pic1',res)
    if cv2.waitKey(1)&0xFF == 27:
        break
cv2.destroyAllWindows()