'''预测类'''

import torch
# from models.UnetModel2 import UNet3D
from os.path import join
import numpy as np
# from skimage import exposure
from models.model import LoadModel
from torch.nn import functional as F

class ModelPredictClass:
    def __init__(self, modelPath, device=torch.device('cpu'), fieldLen=0):
        self.device = device
        # 加载网络
        modelCfg = {
            'name': 'UNet3D',
            # number of input channels to the model
            'in_channels': 16,
            # number of output channels
            'out_channels': 1,
            # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
            'layer_order': 'gcr',
            # number of features at each level of the U-Net
            'f_maps': [16, 32, 64, 128],
            # 'f_maps': [16, 32, 64, 128],
            # 'f_maps_1': [8, 16],
            # 'f_maps_2': [16, 32, 64, 128],
            # 'addMapsId': 1,
            # 'f_maps': [32, 64, 128, 256, 512],
            # number of groups in the groupnorm
            'num_groups': 8,
            # apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax
            # this is only relevant during inference, during training the network outputs logits and it is up to the loss function
            # to normalize with Sigmoid or Softmax
            'final_sigmoid': True,
            # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
            'is_segmentation': True
        }
        self.model = LoadModel(modelCfg, modelPath)
        # self.model = LoadModel(modelCfg)
        self.model.to(device)
        self.model.eval()
        # self.model.train()
        # self.sigmod = torch.nn.Sigmoid()

    def __call__(self, img):
        # img = np.expand_dims(img, axis=0).astype(np.float32)
        # img_eq = exposure.equalize_hist(img, nbins=65536)  # 直方图均衡化
        # img_eq = 1. * img / 65535
        # img = np.array([img, img_eq], dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        # img = np.expand_dims(img_eq, axis=0).astype(np.float32)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = (img - img.mean()) / img.std()
        img = torch.from_numpy(img)
        # img = (img - img.min()) / (img.max() - img.min())
        img = img.to(self.device)
        with torch.no_grad():
            seg = self.model(img)
            # seg = self.sigmod(seg)
            # seg[seg > 0.5] = 255
            seg = seg * 255
            seg = seg.to(torch.uint8).cpu().numpy()[0, 0].astype(np.uint8)
            return seg
