



import cv2
import numpy as np
from matplotlib import pyplot as plt

def cv_show(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('./materials/xiaojiejie.jpg')

# 镜像
mirror = np.fliplr(img)
a = np.hstack((img,mirror))
cv_show('a',a)

# 翻转
flip = np.flipud(img)
b = np.hstack((img,flip))
cv_show('b',b)