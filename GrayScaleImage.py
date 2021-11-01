import numpy as np
import cv2

img = cv2.imread('Water-pause-times.png',0)
img = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
cv2.imshow('image',img)
k = cv2.waitKey(0)&0xFF
if k==27:
  cv2.destroyAllWindows()
elif k == ord('s'):
  cv2.imwrite('123.png',img)
  cv2.destoryAllWindows()