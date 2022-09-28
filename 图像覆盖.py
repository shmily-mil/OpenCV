'''


'''


import numpy as np
import cv2

def cv_show(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('materials/bird.jpg')
print(img.shape)
# img[100:200,100:200] = [255,0,0]
# cv_show('img',img)

img1 = cv2.imread('./materials/skin.jpg')
print(img1.shape)
dst = cv2.resize(img1,(100,100))
img[50:150,50:150] = dst
cv_show('img',img)

# a = cv2.addWeighted(img1,0.5,dst,0.2,0)
# cv_show('a',a)