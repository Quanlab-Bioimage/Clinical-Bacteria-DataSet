'''数据预处理'''
import json
import os, cv2
import shutil, random

import numpy as np
from os.path import join

'''部分数据拷贝'''
def CopyPartData():
    root = r'G:\qxz\MyProject\301Bacteria\DataSet\DataSet2'
    savePath = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\DataSet5000'
    needNum = 5000  # 需要多少数据
    imgType = '.jpg'
    imgPath = join(root, 'images')
    xmlPath = join(root, 'Annotations')
    saveImgPath = join(savePath, 'images')
    saveXmlPath = join(savePath, 'Annotations')
    os.makedirs(saveImgPath, exist_ok=True)
    os.makedirs(saveXmlPath, exist_ok=True)
    xmlLs = os.listdir(xmlPath)
    count = 0
    for name in xmlLs:
        nameId = os.path.splitext(name)[0]
        imgAdd = join(imgPath, nameId + imgType)
        if os.path.isfile(imgAdd):
            shutil.copyfile(imgAdd, join(saveImgPath, nameId + imgType))
            shutil.copyfile(join(xmlPath, name), join(saveXmlPath, name))
            count += 1
            if count > needNum: break

'''Txt合并用于特征监督训练'''
def TxtAddToFetureJson():
    path = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\DataSet1\txt\train.txt'
    savePath = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\DataSet1\txt\fetureTrainInfo.json'
    with open(path, 'r') as f:
        ls = f.read().strip().split('\n')
    nameIdDict = {}
    for name in ls:
        nameId = os.path.splitext(os.path.basename(name))[0]
        bigNameId = nameId.split('_')[0]
        if bigNameId in nameIdDict:
            nameIdDict[bigNameId].append(nameId)
        else:
            nameIdDict[bigNameId] = [nameId]
    with open(savePath, 'w') as f:
        f.write(json.dumps(nameIdDict))

def ShowFeatureLoss():
    from matplotlib import pyplot as plt
    path = './feature.txt'
    data = np.loadtxt(path, ndmin=2)
    plt.plot(data[:, 0])
    plt.plot(data[:, 1])
    plt.show()

'''数据切片'''
def DataSplit():
    newSize = np.array([240, 240])          # 切片大小
    # resizeW, resizeH = 2, 2
    sizeSpace = 5                           # 边缘阈值，在边缘的不要
    newSize2 = newSize - sizeSpace
    newSizeR = (newSize // 2).astype(np.int32)
    # 640数据集地址
    root = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\DataSet3'
    # 小图数据集保存地址
    saveRoot = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\SplitData5'
    # 640数据集下的图像、json、txt文件夹名
    imgPath = join(root, 'images')
    txtPath = join(root, 'labels')
    imgType = '.jpg'                # 640中原始图片的后缀
    txtType = '.txt'                # 640中xml文件的后缀

    saveImgPath = join(saveRoot, 'images')
    saveTxtPath = join(saveRoot, 'labels')
    os.makedirs(saveImgPath, exist_ok=True)
    os.makedirs(saveTxtPath, exist_ok=True)
    imgSet = set([os.path.splitext(name)[0] for name in os.listdir(imgPath)])
    xmlSet = set([os.path.splitext(name)[0] for name in os.listdir(txtPath)])
    nameIdLs = list(imgSet & xmlSet)
    nameIdLs = sorted(nameIdLs)
    for nameId in nameIdLs:
        smallCount = 0
        img = cv2.imread(join(imgPath, nameId + imgType))
        # xml信息，类别框子
        h, w = img.shape[:2]
        tmpInfo = np.loadtxt(join(txtPath, nameId + txtType), ndmin=2)
        # 框子信息，框子中点信息
        newRectJsonInfo = []
        rectInfo, rectCenterLs= [], []
        for item in tmpInfo:
            cx, cy = item[1] * w, item[2] * h
            cw, ch = item[3] * w / 2, item[4] * h / 2
            x0, y0 = int(cx - cw), int(cy - ch)
            x1, y1 = int(cx + cw), int(cy + ch)
            if x0 < 5 or y0 < 5 or w - x0 < 5 or h - y0 < 5 or cw > 60 or ch > 60: continue
            rectInfo.append([x0, y0, x1, y1, int(item[0])])
            # rectCenterLs.append([cx, cy])
            newRectJsonInfo.append([[x0, y0, x1, y1], [cx, cy], int(item[0])])
        if len(newRectJsonInfo) == 0: continue
        # rectCenterLs = np.array(rectCenterLs)
        # # 验证rect和json的匹配结果
        # for rect, _, polyData in newRectJsonInfo:
        #     colorTo = {
        #         1: (0, 255, 0),
        #         2: (0, 0, 255)
        #     }
        #     cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 3)
        #     cv2.fillConvexPoly(img, polyData[1], colorTo[polyData[0]])
        # cv2.imwrite(join(root, 'TmpTest', nameId + '.jpg'), img)
        '''切片
        # 1、找到两个中心点最近的格子，任选一个格子作为起始格子
        # 2、找到与起始格子中心点最近的另一个格子，若合并后框子大小满足要求，则重复（2），否则，切片保存删除格子，继续执行（1）
        1、先找到两个距离最近的格子，判断间隔是否大于阈值
        2、加入一个格子计算新加入后整体的形状是否超过阈值（找到加入后形状最小的格子加入），
            若符合条件则继续加格子，否则切片保存，执行（1）
        '''
        # 计算框子与框子的距离
        while len(newRectJsonInfo):
            rectLen = len(newRectJsonInfo)
            if rectLen == 1:                        # 切下第一个框子
                center = newRectJsonInfo[0][1]
                sp = np.max([center - newSizeR, [0, 0]], axis=0).astype(np.int32)
                ep = np.min([sp + newSize, [h, w]], axis=0).astype(np.int32)
                sp = ep - newSize
                smallImg = img[sp[1]: ep[1], sp[0]: ep[0]].copy()

                newRect = np.array([sp[0], sp[1], ep[0], ep[1]])
                newJsonData = []
                for rect in rectInfo:
                    if (rect[:2] - newRect[:2]).min() > 0 and (newRect[2:] - rect[2: 4]).min() > 0:
                        rect2 = rect[:4].copy()
                        rect2[:2] = rect2[:2] - sp
                        rect2[2:] = rect2[2:] - sp
                        cx = (rect2[0] + rect2[2]) / 2 / newSize[0]
                        cy = (rect2[1] + rect2[3]) / 2 / newSize[1]
                        cw = (rect2[2] - rect2[0]) / newSize[0]
                        ch = (rect[3] - rect[1]) / newSize[1]
                        newJsonData.append([int(rect[-1]), cx, cy, cw, ch])
                saveNameId = nameId + '_' + str(smallCount).zfill(5)
                smallImg = cv2.resize(smallImg, (640, 640))
                cv2.imwrite(join(saveImgPath, saveNameId + '.jpg'), smallImg)
                with open(join(saveTxtPath, saveNameId + '.txt'), 'w') as f:
                    for item in newJsonData:
                        f.write('%d %.8f %.8f %.8f %.8f\n' % (item[0], item[1], item[2], item[3], item[4]))

                smallCount += 1
                newRectJsonInfo.pop(0)
                break
            # 找到两个格子合并后形状最小的两个格子
            minD = 1e6
            ti, tj = -1, -1
            for i in range(rectLen - 1):
                for j in range(i + 1, rectLen):
                    p0 = np.min([newRectJsonInfo[i][0][:2], newRectJsonInfo[j][0][:2]], axis=0)
                    p1 = np.max([newRectJsonInfo[i][0][2:], newRectJsonInfo[j][0][2:]], axis=0)
                    p = p1 - p0
                    if (p < newSize).all():
                        d = np.linalg.norm(p)
                        if d < minD:
                            minD = d
                            ti, tj = i, j
            if ti == -1:
                center = newRectJsonInfo[0][1]
                sp = np.max([center - newSizeR, [0, 0]], axis=0).astype(np.int32)
                ep = np.min([sp + newSize, [h, w]], axis=0).astype(np.int32)
                sp = ep - newSize
                smallImg = img[sp[1]: ep[1], sp[0]: ep[0]].copy()

                newRect = np.array([sp[0], sp[1], ep[0], ep[1]])
                newJsonData = []
                for rect in rectInfo:
                    if (rect[:2] - newRect[:2]).min() > 0 and (newRect[2:] - rect[2: 4]).min() > 0:
                        rect2 = rect[:4].copy()
                        rect2[:2] = rect2[:2] - sp
                        rect2[2:] = rect2[2:] - sp
                        cx = (rect2[0] + rect2[2]) / 2 / newSize[0]
                        cy = (rect2[1] + rect2[3]) / 2 / newSize[1]
                        cw = (rect2[2] - rect2[0]) / newSize[0]
                        ch = (rect[3] - rect[1]) / newSize[1]
                        newJsonData.append([int(rect[-1]), cx, cy, cw, ch])
                saveNameId = nameId + '_' + str(smallCount).zfill(5)
                smallImg = cv2.resize(smallImg, (640, 640))
                cv2.imwrite(join(saveImgPath, saveNameId + '.jpg'), smallImg)
                with open(join(saveTxtPath, saveNameId + '.txt'), 'w') as f:
                    for item in newJsonData:
                        f.write('%d %.8f %.8f %.8f %.8f\n' % (item[0], item[1], item[2], item[3], item[4]))

                # newJsonData = []
                # rect = newRectJsonInfo[0][0]
                # rect[:2] = rect[:2] - sp
                # rect[2:] = rect[2:] - sp
                #
                # newJsonData.append([newRectJsonInfo[0][2], (rect[0] + rect[2]) / 2 / newSize[0],
                #                     (rect[1] + rect[3]) / 2 / newSize[1], (rect[2] - rect[0]) / newSize[0], (rect[3] - rect[1]) / newSize[1]])
                # saveNameId = nameId + '_' + str(smallCount).zfill(5)
                # smallImg = cv2.resize(smallImg, (newSize[0] * resizeW, newSize[1] * resizeH))
                # cv2.imwrite(join(saveImgPath, saveNameId + '.jpg'), smallImg)
                # with open(join(saveTxtPath, saveNameId + '.txt'), 'w') as f:
                #     for item in newJsonData:
                #         f.write('%d %.8f %.8f %.8f %.8f\n' % (item[0], item[1], item[2], item[3], item[4]))

                smallCount += 1
                newRectJsonInfo.pop(0)
                continue
            tLs = [ti]
            curRect = newRectJsonInfo[ti][0].copy()
            while 1:
                minD = 1e6
                tj = -1
                for j in range(rectLen):
                    if j in tLs: continue
                    p0 = np.min([curRect[:2], newRectJsonInfo[j][0][:2]], axis=0)
                    p1 = np.max([curRect[2:], newRectJsonInfo[j][0][2:]], axis=0)
                    p = p1 - p0
                    if (p < newSize2).all():
                        d = np.linalg.norm(p)
                        if d < minD:
                            minD, tj = d, j
                if tj == -1:                        # 切下第一个框子
                    center = np.array([(curRect[0] + curRect[2]) / 2, (curRect[1] + curRect[3]) / 2], dtype=np.int32)
                    sp = np.max([center - newSizeR, [0, 0]], axis=0).astype(np.int32)
                    ep = np.min([sp + newSize, [h, w]], axis=0).astype(np.int32)
                    sp = ep - newSize
                    smallImg = img[sp[1]: ep[1], sp[0]: ep[0]].copy()

                    newRect = np.array([sp[0], sp[1], ep[0], ep[1]])
                    newJsonData = []
                    for rect in rectInfo:
                        if (rect[:2] - newRect[:2]).min() > 0 and (newRect[2:] - rect[2: 4]).min() > 0:
                            rect2 = rect[:4].copy()
                            rect2[:2] = rect2[:2] - sp
                            rect2[2:] = rect2[2:] - sp
                            cx = (rect2[0] + rect2[2]) / 2 / newSize[0]
                            cy = (rect2[1] + rect2[3]) / 2 / newSize[1]
                            cw = (rect2[2] - rect2[0]) / newSize[0]
                            ch = (rect[3] - rect[1]) / newSize[1]
                            newJsonData.append([int(rect[-1]), cx, cy, cw, ch])
                    saveNameId = nameId + '_' + str(smallCount).zfill(5)
                    smallImg = cv2.resize(smallImg, (640, 640))
                    cv2.imwrite(join(saveImgPath, saveNameId + '.jpg'), smallImg)
                    with open(join(saveTxtPath, saveNameId + '.txt'), 'w') as f:
                        for item in newJsonData:
                            f.write('%d %.8f %.8f %.8f %.8f\n' % (item[0], item[1], item[2], item[3], item[4]))

                    # newJsonData = []
                    # for t in tLs:
                    #     rect = newRectJsonInfo[t][0]
                    #     rect[:2] = rect[:2] - sp
                    #     rect[2:] = rect[2:] - sp
                    #     newJsonData.append(
                    #         [newRectJsonInfo[t][2], (rect[0] + rect[2]) / 2 / newSize[0],
                    #          (rect[1] + rect[3]) / 2 / newSize[1], (rect[2] - rect[0]) / newSize[0],
                    #          (rect[3] - rect[1]) / newSize[1]])
                    #
                    # saveNameId = nameId + '_' + str(smallCount).zfill(5)
                    # smallImg = cv2.resize(smallImg, (newSize[0] * resizeW, newSize[1] * resizeH))
                    # cv2.imwrite(join(saveImgPath, saveNameId + '.jpg'), smallImg)
                    # with open(join(saveTxtPath, saveNameId + '.txt'), 'w') as f:
                    #     for item in newJsonData:
                    #         f.write('%d %.8f %.8f %.8f %.8f\n' % (item[0], item[1], item[2], item[3], item[4]))

                    smallCount += 1
                    # 删除
                    newRectJsonInfo2 = []
                    for i, item in enumerate(newRectJsonInfo):
                        if i in tLs: continue
                        newRectJsonInfo2.append(item)
                    newRectJsonInfo = newRectJsonInfo2
                    break
                curRect[:2] = np.min([curRect[:2], newRectJsonInfo[tj][0][:2]], axis=0)
                curRect[2:] = np.max([curRect[2:], newRectJsonInfo[tj][0][2:]], axis=0)
                tLs.append(tj)

'''生成txt'''
def GenerateTxt():
    # 训练集、验证集、测试集比例, 相加不能超过1
    dataSetDivide = [0.7, 0.1, 0.2]
    root = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\SplitData3'
    imgPath = join(root, 'images')
    txtPath = join(root, 'txt')
    txtTypes = ['train.txt', 'val.txt', 'test.txt']
    os.makedirs(txtPath, exist_ok=True)
    ls = os.listdir(imgPath)
    random.shuffle(ls)
    totalLen = int(len(ls))
    space1 = int(dataSetDivide[0] * totalLen)
    space2 = int((dataSetDivide[0] + dataSetDivide[1]) * totalLen)
    space3 = int((dataSetDivide[0] + dataSetDivide[1] + dataSetDivide[2]) * totalLen)
    startSpace = [0, space1, space2]
    endSpace = [space1, space2, space3]
    for sP, eP, name in zip(startSpace, endSpace, txtTypes):
        curLs = ls[sP: eP]
        with open(os.path.join(txtPath, name), 'w') as f:
            for it in curLs:
                f.write('%s\n' % os.path.join(imgPath, it))

'''生成txt2'''
def GenerateTxt2():
    path = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\DataSet3\txt'
    root = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\SplitData5'
    imgType = '.jpg'
    labelPath = join(root, 'labels')
    saveTxtPath = join(root, 'txt')
    datasetNameLs = ['train', 'val', 'test']
    os.makedirs(saveTxtPath, exist_ok=True)
    labelDict = {}
    for name in os.listdir(labelPath):
        name2 = name.split('_')
        nameId = '%s_%s_%s' % (name2[0], name2[1], name2[2])
        if nameId in labelDict: labelDict[nameId].append(name)
        else: labelDict[nameId] = [name]
    for datasetName in datasetNameLs:
        saveData = []
        with open(join(path, datasetName + '.txt'), 'r') as f:
            data = f.read().strip().split('\n')
            for name in data:
                nameId = os.path.splitext(os.path.basename(name))[0]
                if nameId in labelDict:
                    for it in labelDict[nameId]:
                        it2 = os.path.splitext(it)[0] + imgType
                        saveData.append(join(root, 'images', it2))
        with open(join(saveTxtPath, datasetName + '.txt'), 'w') as f:
            for it in saveData:
                f.write(it + '\n')

'''合并img和txt可视化'''
def AddImgAndTxtView():
    root = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\SplitData5'
    imgPath = join(root, 'images')
    txtPath = join(root, 'labels')
    savePath = join(root, 'imagesTxtView')
    imgType = '.jpg'
    os.makedirs(savePath, exist_ok=True)
    ls = os.listdir(txtPath)
    for name in ls:
        nameId = os.path.splitext(name)[0]
        img = cv2.imread(join(imgPath, nameId + imgType))
        w, h = img.shape[:2]
        rectLs = np.loadtxt(join(txtPath, name), ndmin=2)
        for rect in rectLs:
            cx, cy = rect[1] * w, rect[2] * h
            cw, ch = rect[3] * w / 2, rect[4] * h / 2
            x0, y0 = int(cx - cw), int(cy - ch)
            x1, y1 = int(cx + cw), int(cy + ch)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 3)
        cv2.imwrite(join(savePath, nameId + '.png'), img)

'''Txt2生成模式检查'''
def Txt2Check():
    root = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\SplitData5'
    txtPath = join(root, 'txt')
    datasetNameLs = ['train', 'val', 'test']
    dataInfoLs = []
    for datasetName in datasetNameLs:
        curDict = set()
        with open(join(txtPath, datasetName + '.txt'), 'r') as f:
            trainLs = f.read().strip().split('\n')
            for name in trainLs:
                name = os.path.basename(name)
                name2 = name.split('_')
                nameId = '%s_%s_%s' % (name2[0], name2[1], name2[2])
                curDict.add(nameId)
        dataInfoLs.append(curDict)
    newData = list(dataInfoLs[0] & dataInfoLs[1] & dataInfoLs[2])
    print('重合个数：', len(newData))


if __name__ == '__main__':
    # CopyPartData()
    # TxtAddToFetureJson()
    # ShowFeatureLoss()
    DataSplit()
    # AddImgAndTxtView()
    # GenerateTxt()
    # GenerateTxt2()
    # Txt2Check()
