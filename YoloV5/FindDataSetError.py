import os
import shutil
import numpy as np
import cv2
from detect import main, parse_opt

'''预测所有数据，找出预测问题数据，重新人工核验修订'''
def BoxMatchBox(preBox, labBox, thre):
    n1 = len(preBox)
    n2 = len(labBox)
    # if n1 != n2: return False
    if n1 == n2 == 0: return 0
    if n1 == 0: return 0
    if n2 == 0: return 0
    scores = np.zeros([n1, n2], dtype=np.float32)
    for i in range(n1):
        for j in range(n2):
            if preBox[i][0] == labBox[j][0]:
                box1 = preBox[i][1: 5]
                box2 = labBox[j][1: 5]
                x11, y11, x12, y12 = box1[0], box1[1], box1[2], box1[3]
                x21, y21, x22, y22 = box2[0], box2[1], box2[2], box2[3]
                xA = max(x11, x21)
                yA = max(y11, y21)
                xB = min(x12, x22)
                yB = min(y12, y22)
                # 两个框各自的面积
                boxAArea = (x12 - x11) * (y12 - y11)
                boxBArea = (x22 - x21) * (y22 - y21)
                # 重叠面积
                interArea = max(xB - xA, 0) * max(yB - yA, 0)
                # 计算IOU
                scores[i, j] = interArea / (boxAArea + boxBArea - interArea + 1e-7)
    count = 0
    for i in range(n1):
        t = np.argmax(scores[i])
        v = scores[i, t]
        if v > thre:
            count += 1
            scores[:, t] = 0
    return count

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
    # 边缘阈值
    boundBox = 0.01
    # 保存路径
    projectPath = os.path.join(os.getcwd(), 'runs/detext')
    # 名称
    name = 'exp10'
    # 标签文件夹，为数据制作时生成的与图像文件夹同目录下的标签文件夹
    labelPath = r'G:\qxz\MyProject\301Bacteria\DataSet\DataSet2\labels'
    # 模型路径
    weightPath = r'G:\qxz\MyProject\301Bacteria\yolov5-master\runs\train\%s\weights\best.pt' % name
    # 配置文件
    dataPath = r'G:\qxz\MyProject\301Bacteria\yolov5-master\data/MyYamlI6Data.yaml'
    # 尺寸
    imgsz = [640, 640]
    # 置信度阈值
    conf_thres = 0.25
    # iou阈值
    iou_threes = 0.45
    # 颜色列表，要和类别数对上
    colorsLs = {'0': [0, 0, 255], '1': [0, 255, 0]}
    # 显卡设备
    device = '0'
    imgPath = os.path.join(os.path.dirname(labelPath), 'images')
    # 预测
    ProPath = os.path.join(projectPath, name)
    # if os.path.isdir(ProPath): shutil.rmtree(ProPath)
    # 结果路径
    resTxtPath = os.path.join(ProPath, 'labels')
    # if os.path.isdir(resTxtPath): shutil.rmtree(resTxtPath)
    opt = parse_opt()
    opt.weights = weightPath
    opt.source = imgPath
    opt.data = dataPath
    opt.imgsz = imgsz
    opt.conf_thres = conf_thres
    opt.iou_thres = iou_threes
    opt.device = device
    opt.project = projectPath
    opt.name = name
    main(opt)
    # 结果分析
    # saveImgPathOk = os.path.join(ProPath, 'Ok')
    # if os.path.isdir(saveImgPathOk): shutil.rmtree(saveImgPathOk)
    # os.makedirs(saveImgPathOk)

    saveImgPathError = os.path.join(ProPath, 'Error')
    saveImgPathError2 = os.path.join(ProPath, 'ErrorMissingInspection')
    if os.path.isdir(saveImgPathError): shutil.rmtree(saveImgPathError)
    os.makedirs(saveImgPathError)
    if os.path.isdir(saveImgPathError2): shutil.rmtree(saveImgPathError2)
    os.makedirs(saveImgPathError2)

    saveTxtInfoPath = os.path.join(ProPath, 'info.txt')

    needMarkImgPath = os.path.join(ProPath, 'images')
    if os.path.isdir(needMarkImgPath): shutil.rmtree(needMarkImgPath)
    os.makedirs(needMarkImgPath)

    needMarkTXTPath = os.path.join(ProPath, 'txt')
    if os.path.isdir(needMarkTXTPath): shutil.rmtree(needMarkTXTPath)
    os.makedirs(needMarkTXTPath)

    imgsz = imgsz[0]
    labelTotalCount = 0
    preTotalCount = 0
    preOkCount = 0
    falseCount = 0  # 误检
    inspeCount = 0  # 漏检
    imgLs = os.listdir(imgPath)
    imgAddDict = {}
    for item in imgLs:
        tmp = os.path.basename(item)
        imgAddDict[os.path.splitext(tmp)[0]] = os.path.join(imgPath, item)
    ls = os.listdir(labelPath)
    boundBox2 = boundBox * imgsz
    for ii, name in enumerate(ls):
        imgName = os.path.splitext(name)[0]
        if imgAddDict[imgName] is None: continue
        oriImg = cv2.imread(imgAddDict[imgName])
        if not os.path.isfile(os.path.join(resTxtPath, name)): continue
        if not os.path.isfile(os.path.join(labelPath, name)): continue

        preData = ReadYoloTxt(os.path.join(resTxtPath, name), imgsz)
        labData = ReadYoloTxt(os.path.join(labelPath, name), imgsz)

        preData2 = []
        for item in preData:
            if boundBox2 < item[1] < item[3] < imgsz - boundBox2 and boundBox2 < item[2] < item[4] < imgsz - boundBox2:
                preData2.append(item)
        preData = preData2

        labData2 = []
        for item in labData:
            if boundBox2 < item[1] < item[3] < imgsz - boundBox2 and boundBox2 < item[2] < item[4] < imgsz - boundBox2:
                labData2.append(item)
        labData = labData2

        labelTotalCount += len(labData)
        preTotalCount += len(preData)
        okCount = BoxMatchBox(preData, labData, iou_threes)
        preOkCount += okCount
        # preErrorCount += len(preData) - okCount
        # labelErrorCount += len(labData) - okCount
        falseCount += len(preData) - okCount
        inspeCount += len(labData) - okCount
        if okCount == len(preData) and okCount == len(labData):
            pass
            # img = oriImg
            # for box2 in preData:
            #     box = np.array(box2[1: 5], dtype=np.int32)
            #     cls = box2[0]
            #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
            #                   color=colorsLs[cls], thickness=1)
            #     if len(box2) > 5:
            #         cv2.putText(img, '%s %.2f' % (cls, box2[5]), (box[0] + 6, box[1] + 13),
            #                     cv2.FONT_HERSHEY_SIMPLEX,
            #                     0.5, (0, 0, 0), thickness=2)
            #     else:
            #         cv2.putText(img, '%s' % cls, (box[0] + 6, box[1] + 13), cv2.FONT_HERSHEY_SIMPLEX,
            #                     0.5, (0, 0, 0), thickness=2)
            # for box2 in labData:
            #     box = np.array(box2[1: 5], dtype=np.int32)
            #     cls = box2[0]
            #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
            #                   color=colorsLs[cls], thickness=3)
            #     cv2.putText(img, cls, (box[0] + 6, box[3]), cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.5, (0, 0, 0), thickness=2)
            # cv2.imwrite(os.path.join(saveImgPathOk, imgName + '.jpg'), img)
        else:
            shutil.copyfile(imgAddDict[imgName], os.path.join(needMarkImgPath, os.path.basename(imgAddDict[imgName])))
            shutil.copyfile(os.path.join(resTxtPath, name), os.path.join(needMarkTXTPath, name))
            leftImg = oriImg.copy()
            rightImg = oriImg.copy()
            img = oriImg
            leftImg[:, -2:, ...] = 0
            rightImg[:, :2, ...] = 0
            for box2 in preData:
                box = np.array(box2[1: 5], dtype=np.int32)
                cls = box2[0]
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                              color=colorsLs[cls], thickness=1)
                cv2.rectangle(leftImg, (box[0], box[1]), (box[2], box[3]),
                              color=colorsLs[cls], thickness=1)
                if len(box2) > 5:
                    cv2.putText(img, '%s %.2f' % (cls, box2[5]), (box[0] + 6, box[1] + 13),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), thickness=2)
                    cv2.putText(leftImg, '%s %.2f' % (cls, box2[5]), (box[0] + 6, box[1] + 13),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), thickness=2)
                else:
                    cv2.putText(img, '%s' % cls, (box[0] + 6, box[1] + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                                thickness=2)
                    cv2.putText(leftImg, '%s' % cls, (box[0] + 6, box[1] + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), thickness=2)
            for box2 in labData:
                box = np.array(box2[1: 5], dtype=np.int32)
                cls = box2[0]
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                              color=colorsLs[cls], thickness=3)
                cv2.rectangle(rightImg, (box[0], box[1]), (box[2], box[3]),
                              color=colorsLs[cls], thickness=3)
                cv2.putText(img, cls, (box[0] + 6, box[3]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), thickness=2)
                cv2.putText(rightImg, cls, (box[0] + 6, box[3]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), thickness=2)
            img = np.concatenate((leftImg, img, rightImg), axis=1)
            cv2.imwrite(os.path.join(saveImgPathError, imgName + '.jpg'), img)

        if len(preData) < len(labData):
            cv2.imwrite(os.path.join(saveImgPathError2, imgName + '.jpg'), img)

        if ii % 100 == 0:
            print('%d | %d' % (ii, len(ls)))
            print('总标签框数量：%d' % labelTotalCount)
            print('总预测框数量：%d' % preTotalCount)
            print('预测正确框数量：%d' % preOkCount)
            print('误检总框数量：%d' % falseCount)
            print('漏检总框数量：%d\n' % inspeCount)

    with open(saveTxtInfoPath, 'w') as f:
        f.write('总标签框数量：%d\n' % labelTotalCount)
        f.write('总预测框数量：%d\n' % preTotalCount)
        f.write('预测正确框数量：%d\n' % preOkCount)
        f.write('误检总框数量：%d\n' % falseCount)
        f.write('漏检总框数量：%d\n' % inspeCount)
