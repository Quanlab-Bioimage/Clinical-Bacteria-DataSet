'''小图预处理'''
import os, cv2, torch
from os.path import join
from utils.general import non_max_suppression, xyxy2xywh
import numpy as np

'''数据切片'''
def ImgSplit():
    txtPath = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\DataSet3\txt\test.txt'
    savePath = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\DataSet3\testSplit\images'
    imgSize = np.array([160, 160], dtype=np.int32)
    redundanceSize = 80
    os.makedirs(savePath, exist_ok=True)
    with open(txtPath, 'r') as f:
        ls = f.read().strip().split('\n')
        for add in ls:
            name = os.path.splitext(os.path.basename(add))[0]
            image = cv2.imread(add)
            bigSize = np.array(image.shape[:-1])
            sliceNumber = np.ceil((1. * bigSize - imgSize) / (imgSize - redundanceSize)).astype(np.int32) + 1
            for ny in range(sliceNumber[0]):
                for nx in range(sliceNumber[1]):
                    sp = (imgSize - redundanceSize) * [ny, nx]
                    ep = np.min([sp + imgSize, bigSize], axis=0)
                    sp = ep - imgSize
                    sImg = image[sp[0]: ep[0], sp[1]: ep[1]]
                    cv2.imwrite(join(savePath, name + '-%d-%d.jpg' % (nx, ny)), sImg)

'''合并小图预测Txt，暂时只考虑单类别情况'''
def AddSmallPredictTxt():
    path = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\DataSet3\testSplit\PreTxt\labels'
    savePath = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\DataSet3\testSplit\PreTxt\labels2'
    bigSize = np.array([640, 640], dtype=np.int32)
    smallSize = np.array([160, 160], dtype=np.int32)
    redun = 80
    conf_thres = 0.25
    iou_thres = 0.45
    single_cls = True
    max_det = 300
    os.makedirs(savePath, exist_ok=True)
    ls = os.listdir(path)
    dataInfoDict = {}
    scale = [smallSize[0], smallSize[1], smallSize[0], smallSize[1]]
    scale2 = [bigSize[0], bigSize[1], bigSize[0], bigSize[1]]
    for name in ls:
        data = np.loadtxt(join(path, name), ndmin=2)
        tname = os.path.splitext(name)[0].split('-')
        nameId, nx, ny = tname[0], int(tname[1]), int(tname[2])
        sp = [nx, ny] * (smallSize - redun)
        ep = np.min([sp + smallSize, bigSize], axis=0)
        sp = ep - smallSize

        data[:, 1: 5] = data[:, 1: 5] * scale
        data[:, 1: 3] = data[:, 1: 3] + sp
        data[:, 1: 5] = data[:, 1: 5] / scale2
        data = np.c_[data[:, 1: 5], data[:, -1], data[:, 0] + 1]
        if nameId in dataInfoDict:
            dataInfoDict[nameId] += data.tolist()
        else:
            dataInfoDict[nameId] = data.tolist()
    # iou去除重复框
    for nameId in dataInfoDict:
        data = np.array(dataInfoDict[nameId]).reshape([-1, 6])
        data = data[None].astype(np.float32)
        data = torch.from_numpy(data)
        newData = non_max_suppression(data, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=single_cls, max_det=max_det)
        newData[0][:, :4] = xyxy2xywh(newData[0][:, :4])
        newData = newData[0].numpy()
        newData = np.c_[newData[:, 5], newData[:, :4], newData[:, 4]]
        with open(join(savePath, nameId + '.txt'), 'w') as f:
            for item in newData:
                f.write('%d %.8f %.8f %.8f %.8f %.8f\n' % (item[0], item[1], item[2], item[3], item[4], item[5]))

if __name__ == '__main__':
    # ImgSplit()
    AddSmallPredictTxt()
