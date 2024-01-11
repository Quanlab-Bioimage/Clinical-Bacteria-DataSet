#小图拼成大图
import cv2
import os
import numpy as np
redundancy = 40  #冗余
sSize = [640,640]
bigImgPath = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\BigPic/000004.jpg'
bigImg = cv2.imread(bigImgPath)


def ComposImg(addName,addPath,savePath):
    resData = np.zeros_like(bigImg)
    imgsItem = os.listdir(addPath)
    for item in imgsItem:
        if item.split('_')[0]==addName:
            img = cv2.imread(addPath+'/'+item)
            i,j = int(item.split('_')[1]),int(item.split('_')[2][:-4])
            startX,endX = i*(sSize[0]-redundancy),(i+1)*sSize[0]-i*redundancy
            startY,endY = j*(sSize[1]-redundancy),(j+1)*sSize[1]-j*redundancy
            if endX > bigImg.shape[0]:
                startX, endX = bigImg.shape[0]-sSize[0],bigImg.shape[0]
            if endY >bigImg.shape[1]:
                startY, endY = bigImg.shape[1] - sSize[1], bigImg.shape[1]
            resData[startX:endX,startY:endY,:] = img
    cv2.imwrite(savePath,resData)




if __name__ == '__main__':
    path = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\Testexp10\PreTxt\imgLabel/'
    savePath = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\Testexp10\PreTxt/'
    allKey = {}
    names = os.listdir(path)
    for name in names:
        if name.split('_')[0] not in allKey.keys():
            allKey[name.split('_')[0]] = 0

    for key in allKey.keys():
        print(key)
        ComposImg(key,path,savePath+'/'+key+'._res.jpg')


