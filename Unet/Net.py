import os, torch
import shutil
import time

import numpy as np
from torch import nn
from torch.optim import lr_scheduler as lrs

class TakeNotesLoss:
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.id = -1

    def update(self, value):
        self.sum += value
        self.count += 1

    def update2(self):
        tmp = self.sum / self.count
        self.sum = 0
        self.count = 0
        self.id += 1
        return tmp

class Trainer:
    def __init__(self, data_loader, test_loader, model, loss_criterion, optimizer, lr_scheduler, eval_metric, modelPath=None, device=torch.device('cpu'), batchSize=1):
        self.batchSize = batchSize
        self.dataLoader = data_loader
        self.testLoader = test_loader
        self.device = device
        # self.valCount = int(30 / self.batchSize)
        self.valCount = 250
        # 网络
        self.super_net = model
        self.super_net.to(self.device)
        '''加载损失函数优化器学习率等'''
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_metric = eval_metric
        self.modelPath = modelPath
        if self.modelPath is None: self.modelPath = './saved_models/'
        if not os.path.isdir(self.modelPath): os.makedirs(self.modelPath)

    def Train(self, turn=2, writer=None):
        train_small_losses = TakeNotesLoss()
        train_big_losses = TakeNotesLoss()
        evalVal = TakeNotesLoss()
        lastEvalVal = 0
        self.valCount = min(len(self.dataLoader) - 1, self.valCount)
        # self.valCount = 1
        iter_count = 0
        # sigmod = nn.Sigmoid()
        for t in range(turn):
            torch.cuda.empty_cache()
            torch.set_grad_enabled(True)
            self.super_net.train()
            for kk, (img, mask, name) in enumerate(self.dataLoader):
                # if img.shape[0] != self.batchSize: continue
                torch.cuda.empty_cache()
                img = img.to(self.device)
                mask = mask.to(self.device)
                seg = self.super_net(img)
                loss = self.loss_criterion(seg, mask)[0]
                train_small_losses.update(loss.item())
                train_big_losses.update(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (iter_count + 1) % 10 == 0:
                    tmpLoss = train_small_losses.update2()
                    writer.add_scalar('Loss/TrainSmallLoss', tmpLoss, train_small_losses.id)
                    print('TRAIN [Epoch %d | %d] [Proce %d | %d] [Loss %.4f]' % (t, turn, kk, len(self.dataLoader), tmpLoss))
                if (iter_count + 1) % self.valCount == 0:
                    torch.cuda.empty_cache()
                    writer.add_scalar('Loss/TrainBigLoss', train_big_losses.update2(), train_big_losses.id)
                    self.super_net.eval()
                    with torch.no_grad():
                        for kk, (img, mask, name) in enumerate(self.testLoader):
                            # if img.shape[0] != self.batchSize: continue
                            img = img.to(self.device)
                            mask = mask.to(self.device)
                            seg = self.super_net(img)
                            # loss0 = self.loss_criterion(seg0, mask) * 0.5
                            # loss1 = self.loss_criterion(seg1, smallMask) * 0.5
                            # loss = loss0 + loss1
                            # valLoss.update(loss.item())
                            # valLoss1.update(loss0.item())
                            # valLoss2.update(loss1.item())
                            # seg = sigmod(seg)
                            eval = self.eval_metric(seg, mask)
                            evalVal.update(eval)
                        curEvalVal = evalVal.update2()
                        writer.add_scalar('Eval/EvalVal', curEvalVal, evalVal.id)
                        curEvalVal = curEvalVal
                        if curEvalVal > lastEvalVal:
                            print('验证集损失减少,保存模型')
                            s1 = time.time()
                            torch.save({'state_dict': self.super_net.state_dict(), 'param': self.optimizer}, os.path.join(self.modelPath,
                                "supernet_%s.pth" % (str(t).zfill(5))))
                            print('SaveModelTime: ', time.time() - s1)
                            lastEvalVal = curEvalVal
                        s1 = time.time()
                        self.lr_scheduler.step(curEvalVal)
                        lr = self.optimizer.param_groups[0]['lr']
                        writer.add_scalar('TrainParam/Lr', lr, evalVal.id)
                        print(time.time() - s1)
                        print('VAL [Epoch %d | %d] [EvalVal; %.4f] [Lr: %f]' % (t, turn, curEvalVal, lr))
                    torch.cuda.empty_cache()
                    self.super_net.train()
                iter_count += 1
                # if kk > 100:
                #     break
