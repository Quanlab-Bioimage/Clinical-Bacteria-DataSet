from os.path import join
# from skimage import exposure
import numpy as np
from torch import nn as nn
import torch, cv2

def MaxProject(img):
    imgs = []
    for z in range(2):
        for y in range(2):
            for x in range(2):
                imgs.append(img[z::2, y::2, x::2])
    img = np.max(imgs, axis=0)
    return img

class GetMultiTypeMemoryDataSetAndCropQxz:
    def __init__(self, path, txtName, imgType, maskType):
        # self.backThre = 0.3
        self.imgPath = join(path, 'images')
        self.maskPath = join(path, 'mask')
        with open(join(path, txtName), 'r') as f:
            self.ls = f.read().strip().split('\n')
        self.imgType = imgType
        self.maskType = maskType

    def __len__(self): return len(self.ls)

    def __getitem__(self, ind):
        nameId = self.ls[ind]
        img = cv2.imread(join(self.imgPath, nameId + self.imgType))
        img = img.astype(np.float32) / 255.0
        # img = (img - self.mean) / self.std
        img = img.transpose((2, 0, 1))
        # img = img[None]
        img = torch.from_numpy(img)
        # img = torch.as_tensor(img.copy()).float().contiguous(),
        mask = cv2.imread(join(self.maskPath, nameId + self.maskType), 0)
        mask[mask > 2] = 0
        # mask[mask > 1] = 0
        mask = torch.from_numpy(mask).long()
        # mask = torch.as_tensor(mask.copy()).long().contiguous()
        return img, mask, nameId
