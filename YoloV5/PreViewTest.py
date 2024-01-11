'''预测可视化测试'''
import torch
import cv2, os
import numpy as np
from os.path import join
from models.common import DetectMultiBackend
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

rgbLs = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (50, 200, 90),
    (200, 50, 90),
    (90, 50, 200)
]

rgbLs = np.array(rgbLs) / 255

def GetImg(add, color=(114, 114, 114)):
    img = cv2.imread(add)
    # img = cv2.copyMakeBorder(img, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=color)  # add border
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = img[None]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    return img

def GetLabel(labelTxt, imgH, imgW):
    targetLs = np.loadtxt(labelTxt, ndmin=2)
    coor = targetLs[:, 1: 5] * [imgW, imgH, imgW, imgH]
    cx, cy, cw, ch = coor.T
    cww, chh = cw / 2, ch / 2
    x0, y0, x1, y1 = cx - cww, cy - chh, cx + cww, cy + chh
    res = np.c_[x0, y0, x1, y1, np.ones(targetLs.shape[0]), targetLs[:, 0]]
    return res

def FToI(a):
    return int(round(a))

if __name__ == '__main__':
    modelPath = r'./runs/train/exp5/weights/best.pt'
    imgPath = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\DataSet2\images\000003_3_3.jpg'
    labelPath = r'G:\qxz\MyProject\301Bacteria\DataSet\FeatureTestDataSet\DataSet2\labels\000003_3_3.txt'
    # 颜色列表，要和类别数对上
    colorsLs = {'0': [0, 0, 255], '1': [0, 255, 0]}
    imgH, imgW = 640, 640
    device = torch.device('cuda:0')
    conf_thres = 0.25
    iou_thres = 0.45
    max_det = 300
    model = DetectMultiBackend(modelPath, device=device)
    model.eval()
    # model.train()
    img = GetImg(imgPath)
    img = img.to(device, non_blocking=True).float() / 255
    targetLs = GetLabel(labelPath, imgH, imgW)
    with torch.no_grad():
        preds, train_out = model(img)
        # NMS
        preds = non_max_suppression(preds,
                                    conf_thres,
                                    iou_thres,
                                    labels=[],
                                    multi_label=True,
                                    agnostic=False,
                                    max_det=max_det)
        NewPredLs = []
        for si, pred in enumerate(preds):
            # 类别间nms
            tmpRed = pred.detach().cpu().numpy()
            delIds = []
            for i in range(tmpRed.shape[0] - 1):
                if i in delIds: continue
                box1 = tmpRed[i]
                for j in range(i + 1, tmpRed.shape[0]):
                    if j in delIds: continue
                    box2 = tmpRed[j]
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
                    if interArea / (boxAArea + boxBArea - interArea + 1e-7) > 0.95:
                        if box1[4] >= box2[4]: delIds.append(j)
                        else: delIds.append(i)
            newRed = []
            for i in range(tmpRed.shape[0]):
                if not(i in delIds):
                    newRed.append(tmpRed[i])
            newRed = np.array(newRed)
            # # 预测特征图
            # f1 = plt.figure()
            # figLs = [221, 222, 223, 224]
            # featVecLsLs = []
            # for fi, featurePic in enumerate(featureLs):
            #     featurePic = featurePic[si]
            #     fs, fh, fw = featurePic.shape
            #     scaleH, scaleW = imgH / fh, imgW / fw
            #     featVecLs = []
            #     plt.subplot(figLs[fi])
            #     for rri, rect in enumerate(newRed):
            #         x0, y0, x1, y1 = FToI(rect[0] / scaleW), FToI(rect[1] / scaleH), FToI(rect[2] / scaleW), FToI(
            #             rect[3] / scaleH)
            #         x1 = max(x0 + 1, x1)
            #         y1 = max(y0 + 1, y0)
            #         ff = featurePic[:, y0: y1, x0: x1].detach().cpu().numpy()
            #         ff = ff.reshape([fs, -1]).T
            #         for item in ff:
            #             plt.plot(item, c=rgbLs[rri % len(rgbLs)])
            #         featVecLs.append(ff)
            #     featVecLsLs.append(featVecLs)
            # plt.savefig('pre.png')
            # # plt.show()
            # # 标签特征图
            # f2 = plt.figure()
            # figLs = [221, 222, 223, 224]
            # featVecLsLs = []
            # for fi, featurePic in enumerate(featureLs):
            #     featurePic = featurePic[si]
            #     fs, fh, fw = featurePic.shape
            #     scaleH, scaleW = imgH / fh, imgW / fw
            #     featVecLs = []
            #     plt.subplot(figLs[fi])
            #     for rri, rect in enumerate(targetLs):
            #         x0, y0, x1, y1 = FToI(rect[0] / scaleW), FToI(rect[1] / scaleH), FToI(rect[2] / scaleW), FToI(
            #             rect[3] / scaleH)
            #         x1 = max(x0 + 1, x1)
            #         y1 = max(y0 + 1, y0)
            #         ff = featurePic[:, y0: y1, x0: x1].detach().cpu().numpy()
            #         ff = ff.reshape([fs, -1]).T
            #         for item in ff:
            #             plt.plot(item, c=rgbLs[rri % len(rgbLs)])
            #         featVecLs.append(ff)
            #     featVecLsLs.append(featVecLs)
            # plt.savefig('label.png')
            # plt.show()

            NewPredLs.append(newRed)
            img2 = (img[si].detach().cpu().numpy()[::-1].transpose(1, 2, 0) * 255).astype(np.uint8)
            img2 = np.ascontiguousarray(img2)
            for rect in newRed:
                box = np.array(rect[: 4], dtype=np.int32)
                cls = str(int(rect[5]))
                cv2.rectangle(img2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colorsLs[cls], 2)
                cv2.putText(img2, '%s %.2f' % (int(rect[5]), rect[4]), (box[0] + 6, box[1] + 13),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), thickness=2)
            # oriImg = cv2.imread(imgPath)
            # cv2.imshow('ori', oriImg)
            cv2.imshow('', img2)
            cv2.waitKey(0)
    print()
