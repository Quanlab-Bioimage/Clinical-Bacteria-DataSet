from torch.nn import functional as F
import tifffile, torch, os
from os.path import join
import torch.nn as nn
import numpy as np
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
# from sklearn.metrics import precision_recall_curve
from torchmetrics.classification import BinaryRecall, BinaryPrecision
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    # assert input.dim() == 3 or not reduce_batch_first
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

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
        mm1 = (m1 * m1)
        mm2 = (m2 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (mm1.sum(1) + mm2.sum(1) + smooth)
        score = 1 - score.sum() / num
        BCE = F.binary_cross_entropy(probs, targets, reduction='mean')
        return 0.6 * score + 0.4 * BCE

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5
        self.thre = 99. / 255

    def forward(self, predict, target):
        num = predict.size(0)
        pre = predict.view(num, -1)
        pre[pre > self.thre] = 1
        pre[pre <= self.thre] = 0
        tar = target.view(num, -1)
        tar[tar > self.thre] = 1
        tar[tar <= self.thre] = 0
        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        return score

class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""
    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)

class ComputePR(nn.Module):
    def __init__(self, wp, wr, thre):
        super(ComputePR, self).__init__()
        self.wp = wp
        self.wr = wr
        self.thre = thre
        self.pre = BinaryPrecision()
        self.rec = BinaryRecall()

    def forward(self, input, target):
        input[input > self.thre] = 1
        input[input < 1] = 0
        input = input.view(-1)
        target = target.view(-1)
        p = self.pre(input, target)
        r = self.rec(input, target)
        return p, r, p * self.wp + r * self.wr

class LSDLoss(nn.Module):
    def __init__(self):
        super(LSDLoss, self).__init__()
        self.maxThre = 3. / 255
        self.maxThre2 = 103 / 255
        self.maxThre3 = 133 / 255
        # self.bcelog_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        self.sigmod = nn.Sigmoid()

    def forward(self, img, input, mask):
        mask2 = mask > self.maxThre
        mask2_sum = max(1, mask2.sum())
        mask3 = mask > self.maxThre2
        mask3_sum = max(1, mask3.sum())
        mask4 = mask > self.maxThre3

        back1 = input > self.maxThre2
        back1_sum = max(1, back1.sum())

        signWeight = img[mask4]
        if signWeight.size()[0] > 0:
            signWeight = 0.5 - (signWeight - signWeight.min()) / (signWeight.max() - signWeight.min()) * 0.5 + 0.5
        signWeight_sum = max(1, signWeight.sum())
        # bceWeight = self.bcelog_loss(input, mask)
        # bceLoss = bceWeight.mean() + bceWeight[mask2].sum() / mask2_sum + bceWeight[mask3].sum() / mask3_sum + bceWeight[back1].sum() / back1_sum * 0.5

        input = self.sigmod(input)
        l1 = self.l1_loss(input, mask)

        l1Loss = l1.mean() + l1[mask2].sum() / mask2_sum + l1[mask3].sum() / mask3_sum + (l1[mask4] * signWeight).sum() / signWeight_sum + l1[
            back1].sum() / back1_sum * 0.5
        # l1Loss = l1.mean() + l1[mask2].sum() / mask2_sum + l1[mask3].sum() / mask3_sum + l1[back1].sum() / back1_sum * 0.5
        # loss2 = 1 - dice_coeff(input, mask, reduce_batch_first=True)
        return l1Loss, [l1Loss]

class EvalScore(nn.Module):
    def __init__(self):
        super(EvalScore, self).__init__()
        self.maxThre = 3. / 255
        self.maxThre2 = 103 / 255
        self.maxThre3 = 133 / 255
        self.l1_loss = nn.L1Loss(reduction='none')
        self.sigmod = nn.Sigmoid()

    def forward(self, img, input, mask):
        mask2 = mask > self.maxThre
        mask2_sum = max(1, mask2.sum())
        mask3 = mask > self.maxThre2
        mask3_sum = max(1, mask3.sum())
        mask4 = mask > self.maxThre3

        back1 = input > self.maxThre2
        back1_sum = max(1, back1.sum())

        signWeight = img[mask4]
        if signWeight.size()[0] > 0:
            signWeight = 0.5 - (signWeight - signWeight.min()) / (signWeight.max() - signWeight.min()) * 0.5 + 0.5
        signWeight_sum = max(1, signWeight.sum())

        l1 = self.l1_loss(input, mask)
        l1Loss = l1.mean() + l1[mask2].sum() / mask2_sum + l1[mask3].sum() / mask3_sum + \
                 (l1[mask4] * signWeight).sum() / signWeight_sum + l1[back1].sum() / back1_sum * 0.5
        # l1Loss = l1.mean() + l1[mask2].sum() / mask2_sum + (l1[mask3] * signWeight).sum() / signWeight_sum + l1[
        #     back1].sum() / back1_sum * 0.5
        # l1Loss = l1.mean() + l1[mask2].sum() / mask2_sum + l1[mask3].sum() / mask3_sum + l1[back1].sum() / back1_sum * 0.5
        return 1 - l1Loss

class MoreClsDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(MoreClsDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.criterion = nn.CrossEntropyLoss() if self.n_classes > 1 else nn.BCEWithLogitsLoss()
        self.softMax = nn.Softmax(dim=1)

    def forward(self, masks_pred, true_masks):
        loss = self.criterion(masks_pred, true_masks)
        true_masks2 = F.one_hot(true_masks, self.n_classes).permute(0, 3, 1, 2).float()
        masks_pred = self.softMax(masks_pred)
        loss += dice_loss(
            masks_pred[:, 1:],
            true_masks2[:, 1:],
            multiclass=True
        )
        loss += dice_loss(
            masks_pred[:, 0],
            true_masks2[:, 0],
            multiclass=True
        )
        return loss, loss.item()

class MoreClsDiceEval(nn.Module):
    def __init__(self, n_classes):
        super(MoreClsDiceEval, self).__init__()
        self.n_classes = n_classes

    def forward(self, mask_pred, mask_true):
        # convert to one-hot format
        mask_true = F.one_hot(mask_true, self.n_classes).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), self.n_classes).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        dice_score = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
        return dice_score

class MoreClsDiceEval2(nn.Module):
    def __init__(self, n_classes):
        super(MoreClsDiceEval2, self).__init__()
        self.n_classes = n_classes

    def forward(self, mask_pred, mask_true):
        # convert to one-hot format
        mask_true = F.one_hot(mask_true, self.n_classes).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), self.n_classes).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        dice_score = []
        for cls in range(1, self.n_classes):
            t_score = multiclass_dice_coeff(mask_pred[:, [cls]], mask_true[:, [cls]], reduce_batch_first=False).detach().cpu().numpy()
            # dice_score = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            dice_score.append(t_score)
        dice_score.append(multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False).detach().cpu().numpy())
        return dice_score

if __name__ == '__main__':
    maskPath = r'G:\qxz\DataSet\BrainSegDataSet301\DataSet2\OriDataSetSpilt2\mask'
    prePath = r'G:\qxz\DataSet\BrainSegDataSet301\DataSet2\OriDataSetSpilt2\Predict2\result'
    name = '00010_3_0_0.tif'
    # name = '00012_1_1_1.tif'
    mask = tifffile.imread(join(maskPath, name)) / 255.
    mask = np.expand_dims(mask, axis=0).astype(np.float32)
    mask = torch.from_numpy(mask)

    seg = tifffile.imread(join(prePath, name)) / 255.
    seg = np.expand_dims(seg, axis=0).astype(np.float32)
    seg = torch.from_numpy(seg)

    precise_dice_loss = SoftDiceLoss()
    loss = precise_dice_loss(mask, seg)
    print(loss)

    dice = DiceLoss()
    loss = dice(mask, seg)
    print(loss)
