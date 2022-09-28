'''

实现原理:利用HSV颜色范围，利用mask去背景

3种不同的图像分割:
1.颜色分割或阈值分割,颜色分割即阈值分割
2.语义分割
3.边缘检测

API:
cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.inRange(img,lower,upper)
'''

import cv2
import numpy as np

def cv_show(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("materials/bird.jpg")
guassion = cv2.GaussianBlur(img,(5,5),0)
hsv = cv2.cvtColor(guassion,cv2.COLOR_BGR2HSV)
low_blue = np.array([100,43,46])
high_blue = np.array([124,255,255])
mask = cv2.inRange(hsv,low_blue,high_blue)
# cv_show('mask',mask)
res = cv2.bitwise_and(img,img,mask=mask)
cv_show("res",res)

