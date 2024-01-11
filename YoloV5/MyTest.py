import os.path
import shutil, yaml
import warnings
import torch, cv2
import numpy as np

warnings.filterwarnings("ignore")
from val import main, parse_opt

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def BoxMatchBox(preBox, labBox, thre):
    n1 = len(preBox)
    n2 = len(labBox)
    # if n1 != n2: return False
    if n1 == n2 == 0: return True
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
    # 预训练模型位置
    expName = 'exp24'
    weightsPath = r'./runs\train\%s\weights\best.pt' % expName
    # 数据配置文件路径, 若数据变化需要更改数据配置文件对应的目录
    dataPath = r'./data/MyYamlI6Data.yaml'
    task = 'test'
    # 单类别
    single_cls = False
    # 保存地址为project+name
    # project = r'./runs/%s' % task
    project = r'./runs/%s' % 'test'
    # 可为val、test两个值，test为制作数据集目录下txt中的test.txt信息用于测试，val一般不使用
    name = expName
    # 每批大小
    batchSize = 1
    # 1280或者640
    imgsz = 640
    conf_thres = 0.25
    iou_thres = 0.45
    # 颜色列表，要和类别数对上
    colorsLs = {'0': [0, 0, 255], '1': [0, 255, 0], '2': [255, 0, 0], '3': [255, 255, 0]}
    # 使用cpu设备号 cuda device, i.e. 0 or 0,1,2,3 or cpu
    device = '0'
    savePath = os.path.join(project, name)
    resTxtPath = os.path.join(savePath, 'labels')
    if os.path.isdir(resTxtPath): shutil.rmtree(resTxtPath)
    # 图片保存路径
    saveTxtInfoPath = os.path.join(savePath, 'ResImg/info.txt')
    saveImgPathOk = os.path.join(savePath, 'ResImg/Ok')
    saveImgPathError = os.path.join(savePath, 'ResImg/Error')
    saveImgPathError2 = os.path.join(savePath, 'ResImg/ErrorMissingInspection')
    if os.path.isdir(saveImgPathOk): shutil.rmtree(saveImgPathOk)
    os.makedirs(saveImgPathOk)
    if os.path.isdir(saveImgPathError): shutil.rmtree(saveImgPathError)
    os.makedirs(saveImgPathError)
    if os.path.isdir(saveImgPathError2): shutil.rmtree(saveImgPathError2)
    os.makedirs(saveImgPathError2)

    opt = parse_opt()
    opt.data = dataPath
    opt.weights = weightsPath
    opt.batch_size = batchSize
    opt.imgsz = imgsz
    opt.conf_thres = conf_thres
    opt.iou_thres = iou_thres
    opt.task = task
    opt.device = device
    opt.project = project
    opt.name = name
    opt.single_cls = single_cls
    main(opt)
    '''可视化结果'''
    labelTotalCount = 0
    preTotalCount = 0
    preOkCount = 0
    falseCount = 0      # 误检
    inspeCount = 0      # 漏检
    # 标签路径
    with open(dataPath, 'r', encoding='utf-8') as f:
        valTxt = yaml.load(f.read(), Loader=yaml.FullLoader)['%s' % task]
        with open(valTxt, 'r') as f:
            tmp = f.read()
            valInfo = tmp.split('\n')[:-1]
        imgAddDict = {}
        for item in valInfo:
            tmp = os.path.basename(item)
            imgAddDict[os.path.splitext(tmp)[0]] = item
        oriImgPath = os.path.dirname(valInfo[0])
        labelPath = os.path.join(os.path.dirname(oriImgPath), 'labels')
        ls = os.listdir(resTxtPath)
        for name in ls:
            imgName = os.path.splitext(name)[0]
            if not imgName in imgAddDict: continue
            oriImg = cv2.imread(imgAddDict[imgName])
            preData = ReadYoloTxt(os.path.join(resTxtPath, name), imgsz)
            labData = ReadYoloTxt(os.path.join(labelPath, name), imgsz)
            labelTotalCount += len(labData)
            preTotalCount += len(preData)
            okCount = BoxMatchBox(preData, labData, iou_thres)
            preOkCount += okCount
            # preErrorCount += len(preData) - okCount
            # labelErrorCount += len(labData) - okCount
            falseCount += len(preData) - okCount
            inspeCount += len(labData) - okCount
            if okCount == len(preData) and okCount == len(labData):
                img = oriImg
                for box2 in preData:
                    box = np.array(box2[1: 5], dtype=np.int32)
                    cls = box2[0]
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                                  color=colorsLs[cls], thickness=1)
                    if len(box2) > 5:
                        cv2.putText(img, '%s %.2f' % (cls, box2[5]), (box[0] + 6, box[1] + 13),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 0), thickness=2)
                    else:
                        cv2.putText(img, '%s' % cls, (box[0] + 6, box[1] + 13), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 0), thickness=2)
                for box2 in labData:
                    box = np.array(box2[1: 5], dtype=np.int32)
                    cls = box2[0]
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                                  color=colorsLs[cls], thickness=3)
                    cv2.putText(img, cls, (box[0] + 6, box[3]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), thickness=2)
                cv2.imwrite(os.path.join(saveImgPathOk, imgName + '.jpg'), img)
            else:
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
                        cv2.putText(img, '%s' % cls, (box[0] + 6, box[1] + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
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
                # cv2.imwrite(os.path.join(saveImgPathError, imgName + '.jpg'), img)
            # 误检
            if okCount < len(preData):
                cv2.imwrite(os.path.join(saveImgPathError, imgName + '.jpg'), img)
            # 漏检
            if okCount < len(labData):
                cv2.imwrite(os.path.join(saveImgPathError2, imgName + '.jpg'), img)
    with open(saveTxtInfoPath, 'w') as f:
        f.write('总标签框数量：%d\n' % labelTotalCount)
        f.write('总预测框数量：%d\n' % preTotalCount)
        f.write('预测正确框数量：%d\n' % preOkCount)
        f.write('误检总框数量：%d\n' % falseCount)
        f.write('漏检总框数量：%d\n' % inspeCount)


