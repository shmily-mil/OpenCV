


import cv2
import numpy as np
from matplotlib import pyplot as plt

def cv_show(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('./materials/cd1.jpg')
img1 = cv2.resize(img,(1000,600))
# cv_show("img",img1)

gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# cv_show('gray',gray)

thresh,dst = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
# cv_show('dst',dst)

img2 = cv2.GaussianBlur(dst,(5,5),3)
# cv_show('img2',img2)

img3 = cv2.Canny(img2,180,255)
# cv_show('img3',img3)

lines = cv2.HoughLines(img3,1,np.pi/180,180)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img1,(x1,y1),(x2,y2),(255,0,0),4)
cv_show('img1',img1)