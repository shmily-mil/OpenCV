'''

图像融合步骤:
1.图像导入
2.调整图像大小
3.融合图像
4.到处结果

API:
1.cv2.addWeighted(img1,weight1,img2,weight2,gamma)

'''

import cv2

def cv_show(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img1 = cv2.imread('../materials/xiaojiejie.jpg')
img2 = cv2.imread('../materials/skin.jpg')
print(img1.shape)
print(img2.shape)
resize_img1 = cv2.resize(img1,(500,800),interpolation=cv2.INTER_AREA)
resize_img2 = cv2.resize(img2,(500,800),interpolation=cv2.INTER_AREA)
print(resize_img1.shape)
print(resize_img2.shape)
dst = cv2.addWeighted(resize_img1,0.3,resize_img2,0.8,0)
cv_show('dst',dst)




