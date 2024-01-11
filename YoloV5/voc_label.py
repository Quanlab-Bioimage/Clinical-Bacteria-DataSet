# -*- coding: utf-8 -*-
import shutil
import xml.etree.ElementTree as ET
import os
from os import getcwd

sets = ['train', 'val', 'test']
# classes = ["car"]   # 改成自己的类别
classInfo = {'0': 0, '1': 1}
txtPath = r'G:\qxz\MyProject\301Bacteria\DataSet\TestDataSet/'
labelPath = r'G:\qxz\MyProject\301Bacteria\DataSet\TestDataSet\labels/'
imgPath = r'G:\qxz\MyProject\301Bacteria\DataSet\TestDataSet\images/'
xmlPath = r'G:\qxz\MyProject\301Bacteria\DataSet\TestDataSet\resXml/'
savePath = r'G:\qxz\MyProject\301Bacteria\DataSet\TestDataSet\paper_data/'
if os.path.isdir(savePath): shutil.rmtree(savePath)
os.makedirs(savePath)
classes = {'1-chengtuan': 1, '0-buguie': 0, '1-tuyuan': 1, '0': 0, '1-tuoyuan': 1, '1-buguize': 1, '0-buguize': 0, '1': 1, '0-chengtuan': 0}
abs_path = os.getcwd()

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
    return x, y, w, h

def convert_annotation(image_id):
    in_file = open ('%s%s.xml' % (xmlPath, image_id), encoding='UTF-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    try:
        w = int(size.find('width').text)
        h = int(size.find('height').text)
    except:
        return 0
    writeStrLs = []
    for obj in root.iter('object'):
        flag = True
        # difficult = obj.find('difficult').text
        if obj.find('difficult'):
            difficult = float(obj.find('difficult').text)
        else:
            difficult = 0
        cls = obj.find('name').text
        if classes.get(cls) is None: continue
        cls_id = classes[cls]
        # classInfo.add(cls)
        # if cls not in classes or int(difficult) == 1:
        #     continue
        # cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        writeStrLs.append(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    if len(writeStrLs):
        out_file = open('%s%s.txt' % (labelPath, image_id), 'w')
        [out_file.write(item) for item in writeStrLs]
        return 1
    return 0

wd = getcwd()
for image_set in sets:
    if not os.path.exists(labelPath):
        os.makedirs(labelPath)
    image_ids = open('%s%s.txt' % (txtPath, image_set)).read().strip().split()
    list_file = open('%s%s.txt' % (savePath, image_set), 'w')
    for image_id in image_ids:
        if convert_annotation(image_id):
            list_file.write('%s%s.jpg\n' % (imgPath, image_id))
    list_file.close()
print(classInfo)

