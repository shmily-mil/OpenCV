import cv2
import numpy as np

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] #用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# 绘图展示
def cv_show(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#读入模板
img_template = cv2.imread(r"../materials/template.png")
cv_show("template",img_template)

#转换为灰度图
img_gray = cv2.cvtColor(img_template,cv2.COLOR_BGR2GRAY)
cv_show("template_gray",img_gray)

#转换为二值图像
ref = cv2.threshold(img_gray,10,255,cv2.THRESH_BINARY_INV)[1]
cv_show("ref",ref)

#计算轮廓
binary,contours,hierarchy = cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img_template,contours,-1,(0,0,255),3)
cv_show("img_template",img_template)
print(np.array(contours).shape)
contours = sort_contours(contours,method="left-to-right")[0]
digits = {}

for (i,c) in enumerate(contours):
    (x,y,w,h) = cv2.boundingRect(c)
    roi = ref[y:y+h,x:x+w]
    roi = cv2.resize(roi,(57,88))

    #每个数字对应一个模板
    digits[i] = roi

#初始化卷积核
rectKernal = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernal = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#读入输入图像，预处理
cd = cv2.imread(r"../materials/cd.png")
cv_show("cd",cd)
cd = resize(cd,width=300)
cd_gray = cv2.cvtColor(cd,cv2.COLOR_BGR2GRAY)
cv_show("cd_gray",cd_gray)

#礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(cd_gray,cv2.MORPH_TOPHAT,rectKernal)
cv_show("tophat",tophat)

gradx = cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)

gradx = np.absolute(gradx)
(minVal,maxVal) = (np.min(gradx),np.max(gradx))
gradx = (255*(gradx-minVal)/(maxVal-minVal))
gradx = gradx.astype("uint8")

print(np.array(gradx).shape)
cv_show("gradx",gradx)

# 通过闭操作，（先膨胀，在腐蚀），将数字连在一起
gradx = cv2.morphologyEx(gradx,cv2.MORPH_CLOSE,rectKernal)
cv_show("gradx",gradx)
# THRESH_OTUS会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
thresh = cv2.threshold(gradx,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show("thresh",thresh)

# 闭操作
thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernal)
cv_show("thresh",thresh)

# 计算轮廓
thresh,threshCnts,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnts = threshCnts
cur_img = cd.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv_show("cur_img",cur_img)
locs = []

# 遍历轮廓
for (i,c) in enumerate(cnts):# 每一块轮廓长宽比例不一样
    # 计算外接矩形
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w / float(h)

    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if ar > 2.5 and ar < 4.5:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            #符合的留下来
            locs.append((x,y,w,h))

# 将符合的轮廓从左到右排序
locs = sorted(locs,key=lambda x:x[0])
output = []

# 遍历每一个轮廓中的数字
for (i,(gx,gy,gw,gh)) in enumerate(locs):
    groupOutput = []

    #根据坐标提取每一组
    group = cd_gray[gy-5:gy+gh+5,gx-5:gx+gw+5]
    cv_show("group",group)
    #预处理
    group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show("group",group)
    # 计算每一组轮廓
    group,digitCnts,hierarchy = cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = sort_contours(digitCnts,method="left-to-right")[0]

    #遍历每一个轮廓
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的大小
        (x,y,w,h) = cv2.boundingRect(c)
        roi = group[y:y + h,x:x + w]
        roi = cv2.resize(roi,(57,88))
        cv_show("roi",roi)

        # 计算匹配得分
        scores = []

        # 在模板中计算每一个得分
        for (digit,digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF)
            (_,score,_,_) = cv2.minMaxLoc(result)
            scores.append(score)

        # 得到最适合的数
        groupOutput.append(str(np.argmax(scores)))

    # 画出来
    cv2.rectangle(cd,(gx-5,gy-5),(gx+gw+5,gy+gh+5),(0,0,255),1)
    cv2.putText(cd,"".join(groupOutput),(gx,gy-15),cv2.FONT_HERSHEY_COMPLEX,0.65,(0,0,255),2)

    #得到结果
    output.extend(groupOutput)

# 打印结果
print()
cv2.imshow("cd",cd)
cv2.waitKey()