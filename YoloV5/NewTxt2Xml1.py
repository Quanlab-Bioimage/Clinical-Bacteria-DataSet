# -*- coding: utf-8 -*-
import os, shutil
from lxml.etree import Element, SubElement, tostring
from PIL import Image

def mathxyxy(x, y, w, h, image_w, image_h):
    # new_x = x * 2 * image_w #minc + maxc
    # new_y = y * 2 * image_h #minr + maxr
    # new_w = w * image_w
    # new_h = h * image_h
    # maxc = (new_x + new_w)/2
    # maxr = (new_y + new_h)/2
    # minr = maxr - new_h
    # minc = maxc - new_w
    new_x = x * image_w
    new_y = y * image_h
    new_w = w * image_w
    new_h = h * image_h
    xmin = new_x - new_w / 2
    ymin = new_y - new_h / 2
    xmax = new_x + new_w / 2
    ymax = new_y + new_h / 2

    return xmin, ymin, xmax, ymax


def reNormalization(clasEach, imw, imh):
    print(clasEach)
    return mathxyxy(float(clasEach[1].strip()), float(clasEach[2].strip()),
                    float(clasEach[3].strip()), float(clasEach[4].strip()), imw, imh)


# yolo to voc
def txt_xml(img_path, img_name, txt_path, img_txt, xml_path, img_xml, isNor=False):
    # 读取txt的信息
    class_name = ['0', '1', '2', '3']
    clas = []
    names = []
    img = Image.open(os.path.join(img_path, img_name))
    # img = cv2.imread(os.path.join(img_path, img_name))
    imw, imh = img.size
    txt_img = os.path.join(txt_path, img_txt)
    # print(txt_path)
    # print(img_txt)
    # print(txt_img)
    with open(txt_img, "r") as f:
        # next(f)
        for line in f.readlines():
            line = line.strip('\n')
            list = line.split(" ")
            # print(list)
            clas.append(list)
            names.append(int(list[0]))
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'img_cat'
    node_filename = SubElement(node_root, 'filename')
    # 图像名称
    node_filename.text = img_name
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(imw)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(imh)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i in range(len(clas)):
        # print(clas)
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = class_name[names[i]]
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = "Unspecified"
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = "truncated"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_ymax = SubElement(node_bndbox, 'ymax')
        if isNor:
            uu = reNormalization(clas[i], imw, imh)
            print(uu)
            node_xmin.text = str(round(float(uu[0])))
            node_ymin.text = str(round(float(uu[1])))
            node_xmax.text = str(round(float(uu[2])))
            node_ymax.text = str(round(float(uu[3])))
        else:
            node_xmin.text = str(round(float(clas[i][1].strip())))
            node_ymin.text = str(round(float(clas[i][2].strip())))
            node_xmax.text = str(round(float(clas[i][3].strip())))
            node_ymax.text = str(round(float(clas[i][4].strip())))

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    img_newxml = os.path.join(xml_path, img_xml)
    file_object = open(img_newxml, 'wb')
    file_object.write(xml)
    file_object.close()

def txt_xml2(size, txtAdd, saveAdd, isNor=False):
    # 读取txt的信息
    # class_name = ['0', '1']
    clas = []
    names = []
    imh, imw = size[1], size[0]
    txt_img = txtAdd
    # print(txt_path)
    # print(img_txt)
    # print(txt_img)
    with open(txt_img, "r") as f:
        # next(f)
        for line in f.readlines():
            line = line.strip('\n')
            list = line.split(" ")
            # print(list)
            clas.append(list)
            names.append(list[0])
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'img_cat'
    node_filename = SubElement(node_root, 'filename')
    # 图像名称
    node_filename.text = ''
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(imw)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(imh)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i in range(len(clas)):
        # print(clas)
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = names[i]
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = "Unspecified"
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = "truncated"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_ymax = SubElement(node_bndbox, 'ymax')
        # x = round(float(clas[i][1].strip()) * imw)
        # y = round(float(clas[i][2].strip()) * imw)
        # w = float(clas[i][3].strip()) * imw
        # h = float(clas[i][4].strip()) * imw
        # ww = int(w / 2)
        # hh = int(h / 2)
        # x0, x1 = x - ww, x + ww
        # y0, y1 = y - hh, y + hh
        if isNor:
            uu = reNormalization(clas[i], imw, imh)
            print(uu)
            node_xmin.text = str(round(float(uu[0])))
            node_ymin.text = str(round(float(uu[1])))
            node_xmax.text = str(round(float(uu[2])))
            node_ymax.text = str(round(float(uu[3])))
        else:
            node_xmin.text = str(round(float(clas[i][1].strip()) * imw))
            node_ymin.text = str(round(float(clas[i][2].strip()) * imh))
            node_xmax.text = str(round(float(clas[i][3].strip()) * imw))
            node_ymax.text = str(round(float(clas[i][4].strip()) * imh))

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    img_newxml = saveAdd
    file_object = open(img_newxml, 'wb')
    file_object.write(xml)
    file_object.close()

if __name__ == "__main__":
    # 图像文件夹所在位置
    img_path = r"G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet\PreTxt\Fiter\images/"
    # 标注文件夹所在位置
    txt_path = r"G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet\PreTxt\Fiter\labels/"
    # txt转化成xml格式后存放的文件夹
    xml_path = r"G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet\PreTxt\Fiter\Annotations/"
    os.makedirs(xml_path, exist_ok=True)
    for img_name in os.listdir(img_path):
        print(img_name)
        img_xml = img_name.split(".")[0] + ".xml"
        img_txt = img_name[:-4] + ".txt"
        txt_xml(img_path, img_name, txt_path, img_txt, xml_path, img_xml, True)
        # try:
        #     txt_xml(img_path, img_name, txt_path, img_txt, xml_path, img_xml, True)
        # except:
        #     print(img_name[:-4] + ".txt")
        #     # continue
