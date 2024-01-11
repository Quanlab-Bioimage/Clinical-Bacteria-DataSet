import os
import shutil
import xml.etree.ElementTree as ET

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]

'''读取xml'''
def ReadXML(add, forceSize=None):
    in_file = open(add, encoding='UTF-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    if size is None:
        return False, 'No Find Size!%s' % add
    w = size.find('width').text
    if w is None:
        if forceSize is not None and forceSize[0]:
            w = forceSize[1]
        else:
            return False, 'No Find Size-width!%s' % add
    w=int(640)
    h=int(640)
    # w = int(w)
    # h = size.find('height').text
    # if h is None:
    #     if forceSize is not None and forceSize[0]:
    #         h = forceSize[2]
    #     else:
    #         return False, 'No Find Size-height!%s' % add
    # h = int(h)
    dataInfo = []
    for obj in root.iter('object'):
        # try:
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        b1 = float(xmlbox.find('xmin').text)
        b2 = float(xmlbox.find('xmax').text)
        b3 = float(xmlbox.find('ymin').text)
        b4 = float(xmlbox.find('ymax').text)
        bb = convert((w, h), (b1, b2, b3, b4))
        dataInfo.append(bb + [cls, w, h])
        # except Exception as e:
        #     return False, '%s %s' % (add, e)
    return True, dataInfo

'''
通过txt信息复制图像到文件夹
path: 目录
'''
def CopyImgToTxt(path):
    txtTypes = ['train.txt', 'val.txt', 'test.txt']
    for txt in txtTypes:
        if os.path.isfile(path + txt):
            saveAdd = path + os.path.splitext(txt)[0] + 'Img/'
            if os.path.isdir(saveAdd): shutil.rmtree(saveAdd)
            os.makedirs(saveAdd)
            with open(path + txt, 'r') as f:
                while True:
                    add = f.readline().strip()
                    if len(add) == 0: break
                    shutil.copyfile(add, saveAdd + os.path.basename(add))

if __name__ == '__main__':
    path = r'G:\qxz\MyProject\301Bacteria\DataSet\DataSet1\txt/'
    CopyImgToTxt(path)
