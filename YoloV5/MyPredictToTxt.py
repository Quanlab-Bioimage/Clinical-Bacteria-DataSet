import os
import shutil
import numpy as np
import cv2
from detect import main, parse_opt

'''预测图片文件夹，存储为txt'''

def ReadYoloTxt(add, imgsz=1):
    info = []
    with open(add, 'r') as f:
        while True:
            tmp = f.readline().strip()
            if len(tmp) == 0: break
            tmp = tmp.split(' ')
            cx = float(tmp[1]) * imgsz
            cy = float(tmp[2]) * imgsz
            ww = float(tmp[3]) * imgsz / 2
            hh = float(tmp[4]) * imgsz / 2
            if imgsz == 1:
                x0 = float(cx - ww)
                y0 = float(cy - hh)
                x1 = float(cx + ww)
                y1 = float(cy + hh)
            else:
                x0 = int(cx - ww)
                y0 = int(cy - hh)
                x1 = int(cx + ww)
                y1 = int(cy + hh)
            if len(tmp) > 5:
                info.append([tmp[0], x0, y0, x1, y1, float(tmp[5])])
            else:
                info.append([tmp[0], x0, y0, x1, y1])
    return info

if __name__ == '__main__':
    # 图片路径, 默认保存在imgPath同目录下的PreTxt文件夹下
    imgPath = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet\images'
    # 模型路径
    weightPath = r'./runs\train\exp24\weights\best.pt'
    # 配置文件
    dataPath = r'./data/MyYamlI6Data.yaml'
    # 是否保存图片
    isSaveImg = True
    # # 单类别
    # single_cls = True
    # 颜色列表，要和类别数对上
    colorsLs = {'0': [0, 0, 255], '1': [0, 255, 0], '2': [255, 0, 0], '3': [255, 255, 0]}
    txtsz = [640, 640]
    # 尺寸
    imgsz = [640, 640]
    # 置信度阈值
    conf_thres = 0.25
    # iou阈值
    iou_threes = 0.45
    # 显卡设备
    device = '0'
    projectPath = os.path.dirname(imgPath)
    savePath = os.path.join(projectPath, 'PreTxt')
    if os.path.isdir(savePath): shutil.rmtree(savePath)
    opt = parse_opt()
    opt.weights = weightPath
    opt.source = imgPath
    opt.data = dataPath
    opt.imgsz = imgsz
    opt.conf_thres = conf_thres
    opt.iou_thres = iou_threes
    opt.device = device
    opt.project = projectPath
    opt.name = 'PreTxt'
    # opt.single_cls = single_cls
    main(opt)
    # if imgPath:
    #     imgDict = {}
    #     for name in os.listdir(imgPath): imgDict[os.path.splitext(name)[0]] = name
    #     saveImgPath = os.path.join(savePath, 'imgLabel')
    #     if os.path.isdir(saveImgPath): shutil.rmtree(saveImgPath)
    #     os.makedirs(saveImgPath)
    #     saveLabelPath = os.path.join(savePath, 'labels')
    #     ls = os.listdir(saveLabelPath)
    #     for name in ls:
    #         boxLs = ReadYoloTxt(os.path.join(saveLabelPath, name), txtsz[0])
    #         imgName = imgDict.get(os.path.splitext(name)[0])
    #         if imgName is None:
    #             print('Error: ', name)
    #             continue
    #         img = cv2.imread(os.path.join(imgPath, imgName))
    #         for box2 in boxLs:
    #             box = np.array(box2[1: 5], dtype=np.int32)
    #             cls = box2[0]
    #             cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
    #                           color=colorsLs[cls], thickness=7)
    #             if len(box2) > 5:
    #                 cv2.putText(img, '%s %.2f' % (cls, box2[5]), (box[0] + 6, box[1] + 13),
    #                             cv2.FONT_HERSHEY_SIMPLEX,
    #                             0.5, (0, 0, 0), thickness=2)
    #             else:
    #                 cv2.putText(img, '%s' % cls, (box[0] + 6, box[1] + 13), cv2.FONT_HERSHEY_SIMPLEX,
    #                             0.5, (0, 0, 0), thickness=2)
    #         cv2.imwrite(os.path.join(saveImgPath, imgName), img)


