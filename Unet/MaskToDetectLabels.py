'''Mask转检测的框'''

import os, cv2
from os.path import join
import numpy as np
import cc3d

'''Labels转其他类别Labels'''
def LabelsToLabels():
    root = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\SegmentNegPosDataSet'
    txtPath = join(root, 'PRComputeData/labels')
    txtSavePath = join(root, 'PRComputeData/labelsNew')
    labelsToLabels = {
        0: 0,
        1: 1,
        2: 0,
        3: 1,
    }
    os.makedirs(txtSavePath, exist_ok=True)
    ls = os.listdir(txtPath)
    for name in ls:
        data = np.loadtxt(join(txtPath, name), ndmin=2)
        for ti in range(data.shape[0]):
            data[ti, 0] = labelsToLabels[data[ti, 0]]
        np.savetxt(join(txtSavePath, name), data, fmt='%d %.8f %.8f %.8f %.8f')

# Mask And Labels To NewLabels
def MaskLabelsToLabels():
    root = r'G:\qxz\MyProject\301Bacteria\DataSet\PaperDataSet301-V2\SegmentNegPosDataSet'
    task = 'val'
    imgSize = np.array([640, 640], dtype=np.int32)
    maskPath = join(root, 'Predict/%s' % task)
    labelsPath = join(root, 'PRComputeData/labelsNew')
    savePath = join(root, 'PRComputeData/%sLabels' % task)
    keysLs = [1, 2]
    os.makedirs(savePath, exist_ok=True)
    with open(join(root, '%s.txt' % task), 'r') as f:
        idLd = f.read().strip().split('\n')
    # ls = os.listdir(labelsPath)
    for nameId in idLd:
        data = np.loadtxt(join(labelsPath, nameId + '.txt'), ndmin=2)
        maskAdd = join(maskPath, nameId + '.png')
        if not os.path.isfile(maskAdd):
            mask = np.zeros(imgSize, dtype=np.uint8)
        else:
            mask = cv2.imread(maskAdd, 0)
        newData = []
        # mask2 = (mask == 1) * 255
        # labels_out2, label_N2 = cc3d.connected_components(mask2, return_N=True)
        # mask3 = (mask == 2) * 255
        # labels_out3, label_N3 = cc3d.connected_components(mask3, return_N=True)
        for ti, item in enumerate(data):
            cx, cy, w, h = item[1:] * [imgSize[1], imgSize[0], imgSize[1], imgSize[0]]
            ww, hh = w / 2, h / 2
            x0, y0, x1, y1 = int(round(cx - ww)), int(round(cy - hh)), int(round(cx + ww)), int(round(cy + hh))
            x0 = np.clip(x0, 0, imgSize[1])
            y0 = np.clip(y0, 0, imgSize[0])
            x1 = np.clip(x1, 0, imgSize[1])
            y1 = np.clip(y1, 0, imgSize[0])
            sImg = mask[y0: y1, x0: x1]
            cls = 3
            maxN = 0
            for key in keysLs:
                curN = (sImg == key).sum()
                if curN > maxN:
                    cls = key
                    maxN = curN
            if cls == 3:
                continue
            else:
                data[ti, 0] = cls - 1
                newData.append(np.copy(data[ti]))
            mask[y0: y1, x0: x1] = 0
        mask2 = (mask == 1) * 255
        labels_out2, label_N = cc3d.connected_components(mask2, return_N=True)
        for n in range(1, label_N + 1):
            p3d = np.array(np.where(labels_out2 == n)).T
            y0, x0 = np.min(p3d, axis=0)
            y1, x1 = np.max(p3d, axis=0)
            cx, cy = (x0 + x1) / 2 / imgSize[1], (y0 + y1) / 2 / imgSize[0]
            w, h = (x1 - x0), (y1 - y0)
            if w < 10 or h < 10: continue
            w = w / imgSize[1]
            h = h / imgSize[1]
            newData.append(np.array([0, cx, cy, w, h]))
        mask3 = (mask == 2) * 255
        labels_out2, label_N = cc3d.connected_components(mask3, return_N=True)
        for n in range(1, label_N + 1):
            p3d = np.array(np.where(labels_out2 == n)).T
            y0, x0 = np.min(p3d, axis=0)
            y1, x1 = np.max(p3d, axis=0)
            cx, cy = (x0 + x1) / 2 / imgSize[1], (y0 + y1) / 2 / imgSize[0]
            w, h = (x1 - x0), (y1 - y0)
            if w < 10 or h < 10: continue
            w = w / imgSize[1]
            h = h / imgSize[1]
            newData.append(np.array([1, cx, cy, w, h]))
        newData = np.array(newData)
        newData = np.c_[newData, np.ones([newData.shape[0], 1])]
        if newData.shape[0] == 0: continue
        np.savetxt(join(savePath, nameId + '.txt'), newData, fmt='%d %.8f %.8f %.8f %.8f %8.f')

if __name__ == '__main__':
    # LabelsToLabels()
    MaskLabelsToLabels()
