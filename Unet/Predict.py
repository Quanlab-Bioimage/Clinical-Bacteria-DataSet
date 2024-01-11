'''预测'''
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil
import torch
from models.UnetModel2 import UNet3D
# from models import SuperSegM
from os.path import join
import os, tifffile, cv2
import numpy as np
from DataLoader import GetMultiTypeMemoryDataSetAndCropQxz
from LossPy import SmoothL1Loss, LSDLoss, EvalScore, MoreClsDiceLoss, MoreClsDiceEval, MoreClsDiceEval2
from Util import SoftDiceLoss, BCEFocalLoss
from torch.utils.data import DataLoader
from models.model import LoadModel
from torch.nn import functional as F
from torch import nn as nn
from LossPy import BCEDiceLoss, ComputePR

if __name__ == '__main__':
    root = r"G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\SegmentNegPosDataSet"
    task = 'test'
    testTxt = "%s.txt" % task
    modelPath = r'./ModelSave/exp000/supernet_00060.pth'
    savePathMask = join(root, 'Predict/%s' % task)
    savePathMaskView = join(root, 'Predict/%s_View' % task)
    batchSize = 8
    imgSize = np.array([640, 640], dtype=np.int32)
    colorToLabel = {
        1: [0, 255, 0],
        2: [0, 0, 255]
    }
    imgType = '.jpg'
    maskType = '.png'
    device = torch.device('cuda:0')
    dataset = GetMultiTypeMemoryDataSetAndCropQxz(root, testTxt, imgType, maskType)
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=0)

    if os.path.isdir(savePathMask): shutil.rmtree(savePathMask)
    os.makedirs(savePathMask)
    if os.path.isdir(savePathMaskView): shutil.rmtree(savePathMaskView)
    os.makedirs(savePathMaskView)

    # 加载网络
    modelCfg = {
        'name': 'UNet3D',
        # number of input channels to the model
        'in_channels': 3,
        # number of output channels
        'out_channels': 3,
        # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
        'layer_order': 'gcr',
        # number of features at each level of the U-Net
        'f_maps': [16, 32, 64, 128, 256],
        # number of groups in the groupnorm
        'num_groups': 8,
        # apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax
        # this is only relevant during inference, during training the network outputs logits and it is up to the loss function
        # to normalize with Sigmoid or Softmax
        'final_sigmoid': False,
        # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
        'is_segmentation': True
    }
    model = LoadModel(modelCfg, modelPath)
    model.to(device)
    model.eval()
    # 损失
    loss_criterion = MoreClsDiceLoss(modelCfg['out_channels'])
    lsLen = len(loader)
    # 评估
    eval_metric = MoreClsDiceEval2(modelCfg['out_channels'])
    eval_metric.to(device)
    # 保存信息
    # sigmod = nn.Sigmoid()
    scoreLs = []
    for kk, (img, mask, nameLs) in enumerate(loader):
        # if img.shape[0] != batchSize: continue
        img = img.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            seg = model(img)
            eval = eval_metric(seg, mask)
            scoreLs.append(eval)
            # seg = sigmod(seg)
            # loss = loss_criterion(seg, mask)
            mask_pred = seg.argmax(dim=1)
            mask_pred = mask_pred.detach().cpu().numpy()
            for ti, name in enumerate(nameLs):
                nameId = os.path.splitext(name)[0]
                mask = mask_pred[ti].astype(np.uint8)
                cv2.imwrite(join(savePathMask, nameId + '.png'), mask)
                mask2 = np.zeros([mask.shape[0], mask.shape[1], 3], dtype=mask.dtype)
                for key in colorToLabel:
                    mask2[mask == key] = colorToLabel[key]
                cv2.imwrite(join(savePathMaskView, nameId + '.png'), mask2)
            # seg2 = (seg * 255).to(torch.uint8).cpu().numpy()[0, 0]
            # imgName = os.path.splitext(name[0])[0]
            # cv2.imwrite(join(savePath, imgName + '.png'), seg2)
        if kk % 10 == 0:
            print('%d | %d' % (kk, lsLen))
    scoreLs = np.array(scoreLs)
    print(scoreLs)
    print('各类和平均Dice：')
    print(scoreLs.mean(axis=0))

'''
1_2_13_0.tif

'''
