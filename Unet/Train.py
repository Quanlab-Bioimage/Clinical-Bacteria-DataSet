import os, importlib
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from models.model import LoadModel
from torch.utils.data import DataLoader
from DataLoader import GetMultiTypeMemoryDataSetAndCropQxz
import numpy as np
import torch, os
from tensorboardX import SummaryWriter
from Net import Trainer
from MyUtil import GetLossOptimiLr

'''
可视化
cmd
activate xxx
cd logs
tensorboard --logdir "./" --host=0.0.0.0

'''

def Train():
    rootPath = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\SegmentNegPosDataSet'       # 数据路径
    trainTxt = r"train.txt"         # 训练的txt名称
    valTxt = r"val.txt"             # 验证的txt名称
    batchSize = 8
    imgType = '.jpg'
    maskType = '.png'
    device = torch.device('cuda:0')
    logPath = './logs/'             # 日志路径
    if not os.path.isdir(logPath): os.makedirs(logPath)
    logName = len(os.listdir(logPath))
    expName = 'exp%s' % str(logName).zfill(3)
    logAdd = './logs/' + expName
    while True:
        if os.path.isdir(logAdd):
            logName += 1
            logAdd = './logs/exp%s' % str(logName).zfill(3)
        else:
            break
    writer = SummaryWriter(logAdd)
    savePath = r'./ModelSave/%s' % expName          # 模型保存路径
    # 加载数据
    train_dataset = GetMultiTypeMemoryDataSetAndCropQxz(rootPath, trainTxt, imgType, maskType)
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=0)
    val_dataset = GetMultiTypeMemoryDataSetAndCropQxz(rootPath, valTxt, imgType, maskType)
    val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=True, num_workers=0)
    if not os.path.isdir(savePath): os.makedirs(savePath)
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
        # 'f_maps_1': [8, 16],
        # 'f_maps_2': [16, 32, 64, 128],
        # 'addMapsId': 1,
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
    # model = LoadModel(modelCfg, r'./ModelSave/exp006/supernet_00000.pth')
    model = LoadModel(modelCfg)
    model.to(device)
    model.train(True)
    # 获取损失优化器学习率
    loss_criterion, optimizer, lr_scheduler, eval_metric = GetLossOptimiLr(model, modelCfg['out_channels'])
    eval_metric.to(device)
    # 训练
    netObj = Trainer(train_loader, val_loader, model, loss_criterion, optimizer, lr_scheduler, eval_metric, modelPath=savePath, device=device, batchSize=batchSize)
    netObj.Train(turn=500, writer=writer)

if __name__ == '__main__':
    Train()
