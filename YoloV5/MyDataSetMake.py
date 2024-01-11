import os, shutil
import random

from MyUtil import ReadXML

'''生成label数据，txt格式，输入为xml路径'''
def GenerateLabel(xmlPath, labelPath, classInfo, boundThre, smallWThre, smallHThre, ForceSize):
    ls = os.listdir(xmlPath)
    infoDict = []
    noFindTarXml = []
    outRange = []
    imgNameLs = []
    smallBox = []
    for name in ls:
        res, tmpInfo = ReadXML(os.path.join(xmlPath, name), forceSize=ForceSize)
        if not res:
            infoDict.append(tmpInfo)
            continue
        if len(tmpInfo) == 0:
            noFindTarXml.append(name)
            continue
        txtData = []
        for i, item in enumerate(tmpInfo):
            val = classInfo.get(item[4])
            if val is None:
                infoDict.append('%s Not in ClassInfo Name: %s' % (item[4], name))
                continue
            # 边界处理
            if item[2] < smallWThre or item[3] < smallHThre:
                smallBox.append('%s 第%d个框太小' % (name, i))
                continue
            tmpW = item[2] / 2
            tmpH = item[3] / 2
            # if item[0] - tmpW < boundThre or item[1] - tmpH < boundThre or item[0] + tmpW > 1 - boundThre or item[1] + tmpH > 1 - boundThre:

            # if not (boundThre < item[0] < item[0] + tmpW < 1 - boundThre and boundThre < item[1] < item[1] + tmpH < 1 - boundThre):
            #     outRange.append('%s 第%d个框在边缘' % (name, i))
            #     continue
            txtData.append([val, item[0], item[1], item[2], item[3]])
        if len(txtData):
            name2 = os.path.splitext(name)[0]
            with open(os.path.join(labelPath, name2 + '.txt'), 'w') as f:
                for it in txtData:
                    f.write('%s %f %f %f %f\n' % (it[0], *it[1:]))
            imgNameLs.append(name2)
    print('Xml问题信息个数 %d 详细信息如下：' % len(infoDict))
    print(infoDict)
    return imgNameLs, outRange, noFindTarXml, smallBox

'''生成txt'''
def GenerateTxt(imgNameLs, imgPath, txtPath, txtTypes, dataSetDivide):
    imgType = os.path.splitext(os.listdir(imgPath)[0])[-1]
    randImgNameLs = imgNameLs
    random.shuffle(randImgNameLs)
    totalLen = int(len(randImgNameLs))
    space1 = int(dataSetDivide[0] * totalLen)
    space2 = int((dataSetDivide[0] + dataSetDivide[1]) * totalLen)
    space3 = int((dataSetDivide[0] + dataSetDivide[1] + dataSetDivide[2]) * totalLen)
    startSpace = [0, space1, space2]
    endSpace = [space1, space2, space3]
    for sP, eP, name in zip(startSpace, endSpace, txtTypes):
        ls = randImgNameLs[sP: eP]
        with open(os.path.join(txtPath, name), 'w') as f:
            for it in ls:
                f.write('%s\n' % os.path.join(imgPath, it + imgType))

if __name__ == '__main__':
    # 框太小阈值
    smallWThre = 0.01
    smallHThre = 0.01
    # 有些xml没有size信息，是否使用先验xml Size信息已缓解xml确实size信息情况. 第一个参数为是否强制，若强制，则width为第二个参数，height为第三个参数
    ForceSize640 = [True, 640, 640]
    # 边界阈值，若离边界不远则丢掉
    boundThre = -10
    # 训练集、验证集、测试集比例, 相加不能超过1
    dataSetDivide = [0.7, 0.2, 0.1]
    # 图形文件夹地址, 图形文件夹名称必须为images，否则无法训练
    imgPath = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301\images'
    # xml文件夹路径
    xmlPath = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301\Annotations'
    # 转换后数据保存的路径
    savePath = os.path.dirname(imgPath)
    labelPath = os.path.join(savePath, 'labels')
    txtPath = os.path.join(savePath, 'txt')
    txtTypes = ['train.txt', 'val.txt', 'test.txt']
    if os.path.isdir(labelPath): shutil.rmtree(labelPath)
    if os.path.isdir(txtPath): shutil.rmtree(txtPath)
    os.makedirs(labelPath)
    os.makedirs(txtPath)
    # 类型转换，需枚举每个标签值，程序转换给每个标签重命名为对应的值，可多对一、一对一，但不能一对多。键和值必须为字符串类型
    classInfo = {
        # '1-chengtuan': '1',
        # '0-buguie': '0',
        # '1-tuyuan': '1',
           '0': '0',
        # '1-tuoyuan': '1',
        # '1-buguize': '1',
        # '0-buguize': '0',
           '1': '1',

        # '0-chengtuan': '0'
    }
    assert os.path.basename(imgPath) == 'images', '图像文件夹名称必须为images，请修改！'
    for key in classInfo:
        assert type(key) == str,  'classInfo字典每个键对应的值都必须为字符串类型'
        assert type(classInfo[key]) == str, 'classInfo字典每个键对应的值都必须为字符串类型'
    assert dataSetDivide[0] + dataSetDivide[1] + dataSetDivide[2] <= 1, '训练集、验证集、测试集比例, 相加不能超过1'
    imgNameLs, outRange, noFindTarXml, smallBox = GenerateLabel(xmlPath, labelPath, classInfo, boundThre, smallWThre, smallHThre, ForceSize640)
    print('框太小的个数%d 详细情况如下：' % len(smallBox))
    print(smallBox)
    print('框在边缘的个数%d 详细情况如下：' % len(outRange))
    print(outRange)
    print('没有标注框文件个数%d 详细信息如下：' % len(noFindTarXml))
    print(noFindTarXml)
    GenerateTxt(imgNameLs, imgPath, txtPath, txtTypes, dataSetDivide)

