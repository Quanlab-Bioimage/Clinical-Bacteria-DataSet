'''数据集统计'''
import json
import os
from os.path import join
import numpy as np

'''检测Txt数据统计'''
def DetectStatisTxtInfo():
    path = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301\labels'
    ls = os.listdir(path)
    clsDict = {}
    bigPicNameLS = set()
    picNumber = 0
    for name in ls:
        nameId = os.path.splitext(name)[0]
        bigPicNameLS.add(nameId.split('_')[0])
        data = np.loadtxt(join(path, name), ndmin=2)
        for item in data:
            cls = int(item[0])
            if cls in clsDict: clsDict[cls] += 1
            else: clsDict[cls] = 1
        if data.shape[0] > 0: picNumber += 1
    totalRectNumber = 0
    for cls in clsDict:
        totalRectNumber += clsDict[cls]
    print('类别框数量信息: ', clsDict)
    print('总框数量: ', totalRectNumber)
    print('小图数量: ', picNumber)
    print('大图数量： ', len(bigPicNameLS))
    print(bigPicNameLS)

'''分割数据集统计'''
def SegmentStatisInfo():
    root = r'D:\BacterialDataSet\PaperSegmentDataSet\DataSet640'
    labelsInfo = ['G', 'G+']
    imgPath = join(root, 'images')
    jsPath = join(root, 'json')
    txtPath = join(root, 'label')
    imgSet = set([os.path.splitext(name)[0] for name in os.listdir(imgPath)])
    jsSet = set([os.path.splitext(name)[0] for name in os.listdir(jsPath)])
    txtSet = set([os.path.splitext(name)[0] for name in os.listdir(txtPath)])
    idLs = list(imgSet & jsSet & txtSet)
    clsDict = {}
    bigPicNameLS = set()
    picNumber = 0
    for ti, id in enumerate(idLs):
        with open(join(jsPath, id + '.json'), 'r') as f:
            data = json.loads(f.read())
        txtData = np.loadtxt(join(txtPath, id + '.txt'))
        if txtData.shape[0] > 0:
            flag = False
            for item in data['shapes']:
                cls = item['label']
                if cls in labelsInfo:
                    if cls in clsDict: clsDict[cls] += 1
                    else: clsDict[cls] = 1
                    flag = True
            if flag:
                picNumber += 1
                bigPicNameLS.add(id.split('_')[0])
        if ti % 500 == 0:
            print(ti)
    totalRectNumber = 0
    for cls in clsDict:
        totalRectNumber += clsDict[cls]
    print('类别数量信息: ', clsDict)
    print('总数量: ', totalRectNumber)
    print('小图数量: ', picNumber)
    print('大图数量： ', len(bigPicNameLS))
    print(bigPicNameLS)

if __name__ == '__main__':
    DetectStatisTxtInfo()
    # SegmentStatisInfo()
