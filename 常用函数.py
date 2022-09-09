'''
调用函数时，带括号调用的是函数的返回值，而不带括号的函数调用的是函数本身

opencv中知识:
1.图片的载入，cv2.imread(),cv2.imshow(),cv2.waitKey(),cv2.destroyAllWindows(),cv2.cvtColor(),cv2.namedWindow()
2.从摄像头读入，cv2.VideoCapture(),cap.isOpened(),cap.read(),cv2.imshow(),cv2.cvtcolor(),cv2.imshow(),cv2.waitKey(),cap.release(),cv2.destroyAllWindows()
3.读入视频文件，cv2.VideoCapture(),cv2.isOpened()，cap.read(),cv2.cvtcolor(),cv2.imshow(),cv2.cv2.waitKey(),cap,release(),cv2.destroyAllWindows()
4.画直线,cv2.line(img,(start),(end),(color),lineWidth)
5.画矩形,cv2.rectangle(img,(top left corner),(lower right corner),(color),linewidth)
6.画圆,cv2.circle(img,(central coordinate),radius,(color),linewidth)
7.写字,cv2.putText(img,text,(lower left corner),font,linewidth,(color))
8.处理鼠标事件,鼠标回调函数:[i for i in dir(cv2) if "EVENT" in i],draw_circle(event,x,y,flags,param),cv2.namedWindow(),cv2.setMouseCallback()
9.图像的基本操作,img[100,100],img[100,100,0],img[100,100]=[255,255,255],img.shape,img.size,img.dtype,cv2.split(img),img[:,:,0]=0,
10.图像几何变换,cv2.resize(img,(2*width,2*height),cv2.INTER_CUBIC)
11.阈值,cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY),参数:cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV,cv2.THRESH_TRUNC,cv2.THRESH_TOZERO,cv2.THRESH_TOZERO_INV
12.图像过滤和图形平滑(低通滤波),cv2.filter2D(img,depth,k),cv2.blur(img,(3,3)),cv2.GuassianBlur(img,(3,3),0),cv2.medianBLur(img,5),cv2.bilateralFilter(img,9,75,75)
13.形态学,cv2.erode(img,k,iterations),cv2.dilate(img,k,iterations),cv2.morphologyEx(img,cv2.MORPH_CLOSE,k),cv2.morphologyEx(img,cv2.MORPH_OPEN,k),cv2.morphologyEx(img,cv2.MORPH_GRADIENt,k)
14.边缘(高通滤波),cv2.Laplacian(img,cv.CV_64F),cv2.Sobel(img,cv.CV_64F,1,0,ksize=5),cv2.Sobel(img,cv.CV_64F,0,1,ksize=5),cv2.Canny(img,minVal最小阈值,maxVal最大阈值)
15.图像轮廓,img,contours,hierarchy=cv2.findContours(img,mode,ApproximationMode),mode:RETR_EXTERNAL=0,RETR_LIST=1,RETR_CCOMP=2,RETR_TREE=3,ApproximationMode:CHAIN_APPROX_NONE,CHAIN_APPROX_SIMPLE
16.绘制轮廓,cv2.drawContours(img,contours,contourIdx,color,linewidth),contourIdx=1表示绘制所有轮廓
17.图像一维直方图(灰度值),cv2.calcHist(img,channels,mask,histSize,range),plt.hist(img.ravel(),bins,range);plt.show()
18.直方图均衡化,cv2.equalizeHist(img)
19.二维直方图(色相和饱和度),cv2.calcHist([img],[channels],None,[bins],[range]),cv2.imshow(),plt.imshow(dst,interpolation = "nearest"),plt.show()
20.模板匹配,cv2.matchTemplate(img,template,method),minVal,maxVal,minLoc,maxLoc=cv2.minMaxLoc(res)
20.1.6种匹配方式,methods=["cv2.TM_CCOEFF","cv2.TM_CCOEFF_NORMED","cv2.TM_CCORR","cv2.TM_CCORR_NORMED","cv2.TM_SQDIFF","cv2.TM_SQDIFF_NORMED"],如果使用**cv.TM_SQDIFF**作为比较方法，则最小值提供最佳匹配。
21.图像融合,
22.霍夫变换,cv2.HoughLines(binary_img,rho,theta,threshold)
23.图像与操作,
24.背景分离,

'''




