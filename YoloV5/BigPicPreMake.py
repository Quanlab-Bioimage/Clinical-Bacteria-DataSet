'''大图数据处理'''
import os
from os.path import join
import shutil, cv2
import numpy as np

'''复制大图'''
def CopyBigData():
    path = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet\images'
    tarPath = r'P:\DataSet1\BigPic\2022BigPic'
    savePath = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet\BigImages'
    bigNameSet = set()
    ls = os.listdir(path)
    tarDict = {}
    for name in os.listdir(tarPath):
        nameId = os.path.splitext(name)[0]
        tarDict[nameId] = name
    os.makedirs(savePath, exist_ok=True)
    for name in ls:
        nameId = os.path.splitext(name)[0].split('_')[0]
        bigNameSet.add(nameId)
    saveNameLs = []
    for nameId in bigNameSet:
        if nameId in tarDict:
            shutil.copyfile(join(tarPath, tarDict[nameId]), join(savePath, tarDict[nameId]))
            saveNameLs.append(nameId)
    print('BigNameLen: ', len(bigNameSet))
    print('SaveNameLen: ', len(saveNameLs))

'''大图TxtImg可视化'''
def ViewBigImgTxtAndImg():
    imgPath = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet\BigImages'
    txtPath = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet\BigImagesTxt'
    savePath = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet\ViewBigImagesTxt2'
    imgType = '.jpg'
    colorsLs = {'0': [0, 0, 255], '1': [0, 255, 0], '2': [255, 0, 0], '3': [255, 255, 0]}
    os.makedirs(savePath, exist_ok=True)
    ls = os.listdir(txtPath)
    for name in ls:
        # name = '000015.txt'
        nameId = os.path.splitext(name)[0]
        boxLs = np.loadtxt(join(txtPath, name), ndmin=2)
        img = cv2.imread(join(imgPath, nameId + imgType))
        bigSize = np.array(img.shape)
        for box2 in boxLs:
            cls = str(int(box2[0]))
            box = box2[1: 5] * [bigSize[1], bigSize[0], bigSize[1], bigSize[0]]
            box = np.round(box).astype(np.int32)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                          color=colorsLs[cls], thickness=7)

            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                          color=colorsLs[cls], thickness=1)
            # if len(box2) > 5:
            #     cv2.putText(img, '%s %.2f' % (cls, box2[5]), (box[0] + 6, box[1] + 13),
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.5, (0, 0, 0), thickness=2)
            # else:
            cv2.putText(img, '%s' % cls, (box[0] + 6, box[1] + 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), thickness=2)
        cv2.imwrite(join(savePath, nameId + imgType), img)

if __name__ == '__main__':
    # CopyBigData()
    ViewBigImgTxtAndImg()

'''
BigNameLen:  2247
SaveNameLen:  329
'''
