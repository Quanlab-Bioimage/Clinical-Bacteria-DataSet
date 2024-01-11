'''大图推理'''
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import torch
import numpy as np
import os, cv2
from os.path import join

if __name__ == '__main__':
    imgPath = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet\BigImages'
    savePath = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\DetectFourDataSet\BigImagesTxt'
    weightPath = r'./runs\train\exp24\weights\best.pt'
    # 配置文件
    dataPath = r'./data/MyYamlI6Data.yaml'
    # 颜色列表，要和类别数对上
    colorsLs = {'0': [0, 0, 255], '1': [0, 255, 0], '2': [0, 0, 255], '3': [255, 255, 0]}
    # 置信度阈值
    conf_thres = 0.25
    # iou阈值
    iou_thres = 0.45
    # 尺寸
    imgsz = [640, 640]
    smallSize = np.array(imgsz)
    # 冗余
    redun = 40
    max_det = 300
    # 显卡设备
    # device = '0'
    device = torch.device('cuda:0')
    os.makedirs(savePath, exist_ok=True)
    model = DetectMultiBackend(weightPath, device=device, dnn=False, data=dataPath, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    bs = 1  # 只能为1
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # 推理
    ls = os.listdir(imgPath)
    for name in ls:
        # name = '000015.jpg'
        bigImg = cv2.imread(join(imgPath, name))
        bigSize = np.array(bigImg.shape[:2])
        sliceNumber = np.ceil((bigSize - smallSize) / (smallSize - redun)).astype(np.int32) + 1
        preLs = torch.Tensor([])
        for ny in range(sliceNumber[0]):
            for nx in range(sliceNumber[1]):
                sp = (smallSize - redun) * [ny, nx]
                ep = np.min([sp + smallSize, bigSize], axis=0)
                sp = np.min([sp, ep - smallSize], axis=0)
                img = bigImg[sp[0]: ep[0], sp[1]: ep[1]]
                # im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
                im = img.copy()
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous
                im = torch.from_numpy(im).to(model.device)
                im = im / 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                with torch.no_grad():
                    pred = model(im, augment=False, visualize=False)
                pred = pred[0][pred[0][:, :, 4] > conf_thres]
                if pred.size()[0] == 0: continue
                pred[:, 0] += sp[1]
                pred[:, 1] += sp[0]
                if len(preLs) == 0:
                    preLs = pred
                else:
                    preLs = torch.cat([preLs, pred], dim=0)
        if preLs.size()[0] == 0:
            continue
        preLs = preLs[None]
        pred = non_max_suppression(preLs, conf_thres, iou_thres, None, False, max_det=max_det)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            # 去除重叠
            tmpRed = det.detach().cpu().numpy()
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
                    # if interArea / (boxAArea + boxBArea - interArea + 1e-7) > 0.95:
                    if interArea / (min(boxAArea, boxBArea) + 1e-7) > 0.85:
                        if box1[4] >= box2[4]:
                            delIds.append(j)
                        else:
                            delIds.append(i)
            newRed = []
            for i in range(tmpRed.shape[0]):
                if not (i in delIds):
                    newRed.append(tmpRed[i])
            newRed = np.array(newRed)
            res = np.c_[newRed[:, -1], newRed[:, :-1]]
            res[:, 1: -1] = res[:, 1:-1] / [bigSize[1], bigSize[0], bigSize[1], bigSize[0]]
            np.savetxt(join(savePath, os.path.splitext(name)[0] + '.txt'), res, fmt='%d %.8f %.8f %.8f %.8f %.8f')
