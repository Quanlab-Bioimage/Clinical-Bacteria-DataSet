import os, torch
import random

import numpy as np
from torch import nn
from torch import functional as F
import tifffile
from tqdm import tqdm

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        #pt = torch.sigmoid(_input)
        pt = _input
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1-self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        # if self.alpha:
        #     loss = loss * self.alpha
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class BCEFocalLossM(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target,scaler):
        #pt = torch.sigmoid(_input)
        pt = _input




        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1-self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        # if self.alpha:
        #     loss = loss * self.alpha
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):#logits,
        num = targets.size(0)
        smooth = 1

        #probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        mm1=(m1*m1)
        mm2=(m2*m2)

        score = 2. * (intersection.sum(1) + smooth) / (mm1.sum(1) + mm2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class DataPackage:
    def __init__(self, lrDir, hrDir, m = 0, s = 0, p = 0.5):
        self.lrDir = lrDir
        self.hrDir = hrDir
        self.meanVal = m
        self.stdVal = s
        self.prob = p

    def SetMean(self, val):
        self.meanVal = val

    def SetStd(self, val):
        self.stdVal = val

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias)


def default_conv3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias)


def prepare(dev, *args):
    # print(dev)
    device = torch.device(dev)
    if dev == 'cpu':
        device = torch.device('cpu')
    return [a.to(device) for a in args]

def RestoreNetImg(img, mean, max):
    # rImg = (img - self.mean1) / self.std1
    #rImg = np.maximum(np.minimum(img * max + mean, 255), 0)
    rImg = img * max + mean
    maxVal = np.max(rImg)
    minVal = np.min(rImg)
    if maxVal <= minVal:
        rImg *= 0
    else:
        rImg = 255./(maxVal - minVal) * (rImg - minVal)
        rImg = np.maximum(np.minimum(rImg, 255), 0)
    return rImg

def RestoreNetImgV2(img, mean, max):
    # rImg = (img - self.mean1) / self.std1
    #rImg = np.maximum(np.minimum(img * max + mean, 255), 0)
    rImg = img * max + mean
    rImg = np.maximum(np.minimum(rImg, 255), 0)
    return rImg

class WDSRBBlock3D(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(WDSRBBlock3D, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 1
        linear = 0.8
        body.append(
            wn(nn.Conv3d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv3d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv3d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        # res = self.body(x) * self.res_scale
        # res += x
        res = self.body(x) + x
        return res

class ResBlock3D(nn.Module):
    def __init__(self,
                 conv=default_conv3d,
                 n_feats=64,
                 kernel_size=3,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(inplace=True),  # nn.LeakyReLU(inplace=True),
                 res_scale=1):

        super(ResBlock3D, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ConvLayer(nn.Module):
    def __init__(self,
                 inplane = 64,
                 n_feats=32,
                 stride = 1,
                 kernel_size=3,
                 bias=True,
                 bn=nn.BatchNorm3d,
                 padding = 1,
                 act=nn.ReLU(inplace=True),  # nn.LeakyReLU(inplace=True),
                 res_scale=1):

        super(ConvLayer, self).__init__()
        m = []
        m.append(nn.Conv3d(inplane, n_feats,kernel_size = kernel_size,
                           stride = stride,padding = padding, bias=bias))
        if bn is not None:
            m.append(bn(n_feats))
        if act is not None:
            m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        return res

class UpLayer(nn.Module):
    def __init__(self,
                 inplane = 64,
                 n_feats=32,
                 scale_factor=2,
                 bn = nn.BatchNorm3d,
                 act=nn.ReLU(inplace=True)  # nn.LeakyReLU(inplace=True),
                 ):

        super(UpLayer, self).__init__()
        m = []
        m.append(nn.Upsample(scale_factor=scale_factor,mode='trilinear'))

        m.append(nn.Conv3d(in_channels=inplane,out_channels = n_feats,
                           kernel_size=3,padding=3//2 ))
        if bn is not None:
            m.append(bn(n_feats))
        m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        return res


class PixelUpsampler3D(nn.Module):
    def __init__(self,
                 upscale_factor,
                 # conv=default_conv3d,
                 # n_feats=32,
                 # kernel_size=3,
                 # bias=True
                 ):
        super(PixelUpsampler3D, self).__init__()
        self.scaleFactor = upscale_factor

    def _pixel_shuffle(self, input, upscale_factor):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        channels //= upscale_factor[0] * upscale_factor[1] * upscale_factor[2]
        out_depth = in_depth * upscale_factor[0]
        out_height = in_height * upscale_factor[1]
        out_width = in_width * upscale_factor[2]
        input_view = input.contiguous().view(
            batch_size, channels, upscale_factor[0], upscale_factor[1], upscale_factor[2], in_depth,
            in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)

    def forward(self, x):
        # x = self.conv(x)
        up = self._pixel_shuffle(x, self.scaleFactor)
        return up


class GetMultiTypeMemoryDataSetAndCrop:
    def __init__(self, dataList, cropSize, epoch):
        self.dataList:DataPackage = dataList
        self.lrImgList = [[] for x in range(len(self.dataList))]
        self.hrImgList = [[] for x in range(len(self.dataList))]

        self.randProbInteval = [0 for x in range(len(self.dataList) + 1)]
        for k in range(1,len(self.dataList)+1):
            self.randProbInteval[k] = self.dataList[k-1].prob * 100 + self.randProbInteval[k-1]

        self.epoch = epoch

        self.beg = [0, 0, 0]
        self.cropSz = cropSize

        for k in range(len(self.dataList)):
            pack = self.dataList[k]
            lrDir = pack.lrDir
            hrDir = pack.hrDir
            lrFileList = []
            hrFileList = []

            for file in os.listdir(lrDir):
                if file.endswith('.tif'):
                    lrFileList.append(file)

            for file in os.listdir(hrDir):
                if file.endswith('.tif'):
                    hrFileList.append(file)

            for ind in tqdm(range(len(lrFileList))):
                lrName = os.path.join(lrDir,lrFileList[ind])
                hrName = os.path.join(hrDir, hrFileList[ind])
                lrImg = tifffile.imread(lrName)
                hrImg = tifffile.imread(hrName)

                lrImg = np.expand_dims(lrImg, axis=0)
                hrImg = np.expand_dims(hrImg, axis=0)

                self.lrImgList[k].append(lrImg)
                self.hrImgList[k].append(hrImg)

    def __len__(self):
        return self.epoch#len(self.hrFileList)

    def len(self):
        return self.epoch#len(self.hrFileList)

    def __getitem__(self, ind):
        flag = True
        dataID = 0
        randNum = np.random.randint(self.randProbInteval[-1])#len(self.dataList)
        for k in range(len(self.randProbInteval)-1):
            if self.randProbInteval[k] < randNum < self.randProbInteval[k + 1]:
                dataID = k
                break

        ind = np.random.randint(len(self.lrImgList[dataID]))
        tryNum = 0
        while flag:
            sz = self.lrImgList[dataID][ind].shape
            self.beg[0] = np.random.randint(0, sz[1] - self.cropSz[0] - 1)
            self.beg[1] = np.random.randint(0, sz[2] - self.cropSz[1] - 1)
            self.beg[2] = np.random.randint(0, sz[3] - self.cropSz[2] - 1)

            hrImg = self.hrImgList[dataID][ind][:, self.beg[0]:self.beg[0] + self.cropSz[0],
                        self.beg[1]:self.beg[1] + self.cropSz[1],
                        self.beg[2]:self.beg[2] + self.cropSz[2]]

            if np.sum(hrImg) < 800 and tryNum < 10:
                tryNum += 1
            else:
                lrImg = self.lrImgList[dataID][ind][:, self.beg[0]:self.beg[0] + self.cropSz[0],
                        self.beg[1]:self.beg[1] + self.cropSz[1],
                        self.beg[2]:self.beg[2] + self.cropSz[2]]
                flag = False


        lrImg = torch.from_numpy(lrImg.copy().astype(np.float)).float()
        hrImg = torch.from_numpy(hrImg.copy().astype(np.float)).float()
        # lrImg = (lrImg - 0*self.dataList[dataID].meanVal) / self.dataList[dataID].stdVal
        hrImg = hrImg / 8.
        return lrImg,  hrImg , self.dataList[dataID].meanVal, self.dataList[dataID].stdVal





import os, torch
import numpy as np
from torch import nn
from torch import functional as F
import tifffile
from tqdm import tqdm
from os.path import join

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        #pt = torch.sigmoid(_input)
        pt = _input
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1-self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        # if self.alpha:
        #     loss = loss * self.alpha
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class BCEFocalLossM(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target,scaler):
        #pt = torch.sigmoid(_input)
        pt = _input
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1-self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        # if self.alpha:
        #     loss = loss * self.alpha
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


# Dice系数
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# class SoftDiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(SoftDiceLoss, self).__init__()
#
#     def forward(self, probs, targets):#logits,
#         num = targets.size(0)
#         smooth = 1
#         #probs = F.sigmoid(logits)
#         m1 = probs.view(num, -1)
#         m2 = targets.view(num, -1)
#         intersection = (m1 * m2)
#         mm1=(m1*m1)
#         mm2=(m2*m2)
#         score = 2. * (intersection.sum(1) + smooth) / (mm1.sum(1) + mm2.sum(1) + smooth)
#         score = 1 - score.sum() / num
#         return score
#         # BCE = F.binary_cross_entropy(probs, targets, reduction='mean')
#         # return 0.6 * score + 0.4 * BCE

def dice_loss(input, target, weight=None):
    thre = 50. / 255
    smooth = 1.0
    loss = 0.0
    iflat = input[:, 0].view(-1).clone()
    tflat = target[:, 0].view(-1).clone()
    iflat[iflat < thre] = 0
    iflat[iflat > 0] = 1
    tflat[tflat < thre] = 0
    tflat[tflat > 0] = 1
    intersection = (iflat * tflat).sum()
    loss += 1 - (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return loss

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):#logits,
        num = targets.size(0)
        smooth = 1
        #probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        mm1=(m1*m1)
        mm2=(m2*m2)
        score = 2. * (intersection.sum(1) + smooth) / (mm1.sum(1) + mm2.sum(1) + smooth)
        score = 1 - score.sum() / num
        score2 = dice_loss(probs, targets)
        return score + score2, score, score2

class DataPackage:
    def __init__(self, lrDir, hrDir, m = 0, s = 0, p = 0.5):
        self.lrDir = lrDir
        self.hrDir = hrDir
        self.meanVal = m
        self.stdVal = s
        self.prob = p

    def SetMean(self, val):
        self.meanVal = val

    def SetStd(self, val):
        self.stdVal = val


class DataPackage1:
    def __init__(self, lrDir, hrDir,hrDir122,hrDir222, p = 0.5):
        self.lrDir = lrDir
        self.hrDir = hrDir
        self.hrDir122 = hrDir122
        self.hrDir222 = hrDir222
        self.prob = p

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias)


def default_conv3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias)


def prepare(dev, *args):
    # print(dev)
    device = torch.device(dev)
    if dev == 'cpu':
        device = torch.device('cpu')
    return [a.to(device) for a in args]

def RestoreNetImg(img, mean, max):
    # rImg = (img - self.mean1) / self.std1
    #rImg = np.maximum(np.minimum(img * max + mean, 255), 0)
    rImg = img * max + mean
    maxVal = np.max(rImg)
    minVal = np.min(rImg)
    if maxVal <= minVal:
        rImg *= 0
    else:
        rImg = 255./(maxVal - minVal) * (rImg - minVal)
        rImg = np.maximum(np.minimum(rImg, 255), 0)
    return rImg

def RestoreNetImgV2(img, mean, max):
    # rImg = (img - self.mean1) / self.std1
    #rImg = np.maximum(np.minimum(img * max + mean, 255), 0)
    rImg = img * max + mean
    rImg = np.maximum(np.minimum(rImg, 255), 0)
    return rImg

class WDSRBBlock3D(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(WDSRBBlock3D, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 1
        linear = 0.8
        body.append(
            wn(nn.Conv3d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv3d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv3d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        # res = self.body(x) * self.res_scale
        # res += x
        res = self.body(x) + x
        return res

class ResBlock3D(nn.Module):
    def __init__(self,
                 conv=default_conv3d,
                 n_feats=64,
                 kernel_size=3,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(inplace=True),  # nn.LeakyReLU(inplace=True),
                 res_scale=1):

        super(ResBlock3D, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ConvLayer(nn.Module):
    def __init__(self,
                 inplane = 64,
                 n_feats=32,
                 stride = 1,
                 kernel_size=3,
                 bias=True,
                 bn=nn.BatchNorm3d,
                 padding = 1,
                 act=nn.ReLU(inplace=True),  # nn.LeakyReLU(inplace=True),
                 res_scale=1):

        super(ConvLayer, self).__init__()
        m = []
        m.append(nn.Conv3d(inplane, n_feats,kernel_size = kernel_size,
                           stride = stride,padding = padding, bias=bias))
        if bn is not None:
            m.append(bn(n_feats))
        if act is not None:
            m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        return res

class UpLayer(nn.Module):
    def __init__(self,
                 inplane = 64,
                 n_feats=32,
                 scale_factor=2,
                 bn = nn.BatchNorm3d,
                 act=nn.ReLU(inplace=True)  # nn.LeakyReLU(inplace=True),
                 ):

        super(UpLayer, self).__init__()
        m = []
        m.append(nn.Upsample(scale_factor=scale_factor,mode='trilinear'))

        m.append(nn.Conv3d(in_channels=inplane,out_channels = n_feats,
                           kernel_size=3,padding=3//2 ))
        if bn is not None:
            m.append(bn(n_feats))
        m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        return res


class PixelUpsampler3D(nn.Module):
    def __init__(self,
                 upscale_factor,
                 # conv=default_conv3d,
                 # n_feats=32,
                 # kernel_size=3,
                 # bias=True
                 ):
        super(PixelUpsampler3D, self).__init__()
        self.scaleFactor = upscale_factor

    def _pixel_shuffle(self, input, upscale_factor):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        channels //= upscale_factor[0] * upscale_factor[1] * upscale_factor[2]
        out_depth = in_depth * upscale_factor[0]
        out_height = in_height * upscale_factor[1]
        out_width = in_width * upscale_factor[2]
        input_view = input.contiguous().view(
            batch_size, channels, upscale_factor[0], upscale_factor[1], upscale_factor[2], in_depth,
            in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)

    def forward(self, x):
        # x = self.conv(x)
        up = self._pixel_shuffle(x, self.scaleFactor)
        return up


class GetMultiTypeMemoryDataSetAndCropM:
    def __init__(self, dataList, cropSize, epoch):
        self.dataList:DataPackage = dataList
        self.lrImgList = [[] for x in range(len(self.dataList))]
        self.hrImgList = [[] for x in range(len(self.dataList))]
        self.hrImgList122 = [[] for x in range(len(self.dataList))]
        self.hrImgList222 = [[] for x in range(len(self.dataList))]

        self.randProbInteval = [0 for x in range(len(self.dataList) + 1)]
        for k in range(1,len(self.dataList)+1):
            self.randProbInteval[k] = self.dataList[k-1].prob * 100 + self.randProbInteval[k-1]

        self.epoch = epoch

        self.beg = [0, 0, 0]
        self.cropSz = cropSize

        for k in range(len(self.dataList)):
            pack = self.dataList[k]
            lrDir = pack.lrDir
            hrDir = pack.hrDir
            hrDir122=pack.hrDir122
            hrDir222 = pack.hrDir222
            lrFileList = []
            hrFileList = []
            hr122FileList = []
            hr222FileList = []

            for file in os.listdir(lrDir):
                if file.endswith('.tif'):
                    lrFileList.append(file)

            for file in os.listdir(hrDir):
                if file.endswith('.tif'):
                    hrFileList.append(file)

            for file in os.listdir(hrDir122):
                if file.endswith('.tif'):
                    hr122FileList.append(file)
            for file in os.listdir(hrDir222):
                if file.endswith('.tif'):
                    hr222FileList.append(file)

            for ind in tqdm(range(len(lrFileList))):
                lrName = os.path.join(lrDir,lrFileList[ind])
                hrName = os.path.join(hrDir, hrFileList[ind])
                hrName122 = os.path.join(hrDir122, hr122FileList[ind])
                hrName222 = os.path.join(hrDir222, hr222FileList[ind])
                lrImg = tifffile.imread(lrName)
                hrImg = tifffile.imread(hrName)
                hrImg122 = tifffile.imread(hrName122)
                hrImg222 = tifffile.imread(hrName222)

                lrImg = np.expand_dims(lrImg, axis=0)
                hrImg = np.expand_dims(hrImg, axis=0)
                hrImg122 = np.expand_dims(hrImg122, axis=0)
                hrImg222 = np.expand_dims(hrImg222, axis=0)

                self.lrImgList[k].append(lrImg)
                self.hrImgList[k].append(hrImg)
                self.hrImgList122[k].append(hrImg122)
                self.hrImgList222[k].append(hrImg222)

    def __len__(self):
        return self.epoch#len(self.hrFileList)

    def len(self):
        return self.epoch#len(self.hrFileList)

    def __getitem__(self, ind):
        flag = True
        dataID = 0
        randNum = np.random.randint(self.randProbInteval[-1])#len(self.dataList)
        for k in range(len(self.randProbInteval)-1):
            if self.randProbInteval[k] < randNum < self.randProbInteval[k + 1]:
                dataID = k
                break

        ind = np.random.randint(len(self.lrImgList[dataID]))
        tryNum = 0
        while flag:
            sz = self.lrImgList[dataID][ind].shape
            self.beg[0] = np.random.randint(0, sz[1] - self.cropSz[0] - 1)
            self.beg[1] = np.random.randint(0, sz[2] - self.cropSz[1] - 1)
            self.beg[2] = np.random.randint(0, sz[3] - self.cropSz[2] - 1)

            hrImg = self.hrImgList[dataID][ind][:, self.beg[0]:self.beg[0] + self.cropSz[0],
                        self.beg[1]:self.beg[1] + self.cropSz[1],
                        self.beg[2]:self.beg[2] + self.cropSz[2]]

            hrImg122 = self.hrImgList122[dataID][ind][:, self.beg[0]:self.beg[0] + self.cropSz[0],
                    self.beg[1]*2:self.beg[1]*2 + self.cropSz[1]*2,
                    self.beg[2]*2:self.beg[2]*2 + self.cropSz[2]*2]

            hrImg222 = self.hrImgList222[dataID][ind][:, self.beg[0]*2:self.beg[0]*2 + self.cropSz[0]*2,
                       self.beg[1] * 2:self.beg[1] * 2 + self.cropSz[1] * 2,
                       self.beg[2] * 2:self.beg[2] * 2 + self.cropSz[2] * 2]

            if np.sum(hrImg) < 800 and tryNum < 10:
                print('信号小于800')
                tryNum += 1
            else:
                lrImg = self.lrImgList[dataID][ind][:, self.beg[0]:self.beg[0] + self.cropSz[0],
                        self.beg[1]:self.beg[1] + self.cropSz[1],
                        self.beg[2]:self.beg[2] + self.cropSz[2]]
                flag = False


        lrImg = torch.from_numpy(lrImg.copy().astype(np.float)).float()
        hrImg = torch.from_numpy(hrImg.copy().astype(np.float)).float()
        hrImg122 = torch.from_numpy(hrImg122.copy().astype(np.float)).float()
        hrImg222 = torch.from_numpy(hrImg222.copy().astype(np.float)).float()
        # lrImg = (lrImg - 0*self.dataList[dataID].meanVal) / self.dataList[dataID].stdVal
        hrImg = hrImg / 8.
        hrImg122 = hrImg122 / 8.
        hrImg222 = hrImg222 / 8.
        return lrImg,  hrImg, hrImg122, hrImg222

class GetMultiTypeMemoryDataSetAndCropQxz:
    def __init__(self, path, txtName, imgSize):
        self.imgSize = imgSize
        self.imgPath = join(path, 'images')
        self.maskPath = join(path, 'mask')
        with open(join(path, txtName), 'r') as f:
            self.nameLs = f.read().strip().split('\n')

    def __len__(self): return len(self.nameLs)

    def __getitem__(self, ind):
        # self.nameLs[ind] = '0000000158_3_2_0.tif'
        img = tifffile.imread(join(self.imgPath, self.nameLs[ind]))[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]]
        img = np.expand_dims(img, axis=0).astype(np.float32)
        mask = tifffile.imread(join(self.maskPath, self.nameLs[ind]))[:self.imgSize[2], :self.imgSize[1], :self.imgSize[0]]
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        # mask = torch.from_numpy(1. * mask / 255)
        mask = torch.from_numpy(0.99 * mask / mask.max())
        # mask = torch.from_numpy(1. * mask)
        # mask[mask > 0] = 0.99
        return img, mask, self.nameLs[ind]
