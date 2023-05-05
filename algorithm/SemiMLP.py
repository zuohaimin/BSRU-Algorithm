import math
from abc import ABC

import torch
from torch import optim
from torch.utils.data import DataLoader

from algorithm.SklearnInterface import SklearnInterface
from algorithm.Sampler import TwoStreamBatchSampler
from algorithm.ramps import sigmoid_rampup
from algorithm.NeuralNetworks import MLPDropoutReg
from algorithm.SemiDataset import SemiDataset
from dataset.ArffRegDataset import ArffRegDataset
from util import RegUtil


class SemiMLP(SklearnInterface, ABC):
    def __init__(self, n_epoch, lr, consistency_scale, consistency_rampup, baseModel, evalData,
                 batch_size=50, label_batch_size=10, verbose=False, ablation=False, addGaussianNoise=True,
                 noise_scale=0.001) -> None:
        super().__init__()
        self.verbose = verbose
        self.label_batch_size = label_batch_size
        self.consistency_rampup = consistency_rampup
        self.epochs = n_epoch
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
        # 提供一致性损失的模型
        self.baseModel = baseModel
        self.consistency_scale = consistency_scale
        # 验证集数据
        self.evalData = evalData
        self.ablation = ablation
        # 添加高斯噪声
        self.addGaussianNoise = addGaussianNoise
        self.noise_scale = noise_scale

    def fit(self, X_l, y_l, X_u):
        # 拟合半监督回归器
        self.baseModel.fit(X_l, y_l, X_u)
        # 构造trainLoader
        self.model = MLPDropoutReg.getDefaultModel(X_l.shape[1], 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        # TODO 添加一致性损失
        # lossFunc = nn.MSELoss(reduction="sum").cuda()
        lossFunc = torch.nn.SmoothL1Loss().cuda()
        semiDataset = SemiDataset(X_l, y_l, X_u, addGaussianNoise=self.addGaussianNoise, noise_scale=self.noise_scale)
        labelIdx, unlabelIdx = semiDataset.getSemiIndex()
        sampler = TwoStreamBatchSampler(primary_indices=unlabelIdx,
                                        secondary_indices=labelIdx,
                                        batch_size=self.batch_size,
                                        secondary_batch_size=self.label_batch_size)
        # semiDataset.to(device)
        dataLoader = DataLoader(semiDataset, batch_sampler=sampler)
        return self.train(self.epochs, self.model, dataLoader, optimizer, lossFunc, self.baseModel, device)

    def train(self, n_epoch, model, dataLoader, optimizer, lossFunc, baseModel, device):
        trainRmseList = []
        trainBatchIdxList = []
        evalRmseList = []
        evalBatchIdxList = []
        for epoch in range(n_epoch):
            if self.verbose:
                print('Epoch [{}/{}]\n'.format(epoch + 1, n_epoch))
            model.train()
            sum_loss = 0.0
            total = 0.0
            for batch_idx, (trains, labels) in enumerate(dataLoader):
                # 一共有多少个batch
                length = len(dataLoader)
                trains = trains.to(device)
                labels = labels.to(device)
                # 梯度清零
                optimizer.zero_grad()
                # 模型正向，获取预测值
                predicts = model(trains)
                labelMask = labels > 0
                unlabelMask = labels == -1
                # 有监督损失
                supervise_loss = lossFunc(predicts[labelMask].ravel(), labels[labelMask])
                if self.ablation:
                    loss = supervise_loss
                else:
                    # 无监督损失
                    pseudoLabel = baseModel.predict(trains[unlabelMask].cpu())
                    pseudoLabelTensor = torch.Tensor(pseudoLabel).to(device)
                    unsupervise_loss = lossFunc(predicts[unlabelMask].ravel(), pseudoLabelTensor)
                    # loss = lossFunc(predicts.squeeze(dim=1), labels)
                    consistency_weight = self.consistency_scale * sigmoid_rampup(epoch, self.consistency_rampup)
                    # consistency_weight = self.consistency_scale * sigmoid_rampup(epoch, n_epoch)
                    loss = supervise_loss + consistency_weight * unsupervise_loss
                loss.backward()
                optimizer.step()
                if self.verbose:
                    # Tensor.item() 类型转换，返回一个数
                    sum_loss += supervise_loss.item()
                    # 注意这里是以一个batch为一个单位
                    # print("[epoch:%d, iter:%d] RMSE Loss: %.03f"
                    #       % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1)))
                    total += labels[labelMask].size(0)
                    totalBatchIdx = batch_idx + 1 + epoch * length
                    trainRmse = math.sqrt(sum_loss / total)
                    print("[epoch:%d, iter:%d] Loss: %.09f"
                          % (epoch + 1, totalBatchIdx, trainRmse))
                    trainBatchIdxList.append(totalBatchIdx)
                    trainRmseList.append(trainRmse)
                    if totalBatchIdx % 10 == 0:
                        # 在测试集上评估
                        evalRmse = RegUtil.evalNNLoss(model, self.evalData, device)
                        # 切换为训练模式
                        evalRmseList.append(evalRmse)
                        evalBatchIdxList.append(totalBatchIdx)
                        model.train()
        if self.verbose:
            RegUtil.plotRmseLoss(trainBatchIdxList, trainRmseList, evalBatchIdxList, evalRmseList)
            return trainBatchIdxList, trainRmseList, evalBatchIdxList, evalRmseList

    def predict(self, X):
        self.model.eval()
        tensorX = torch.Tensor(X).cuda()
        return self.model(tensorX).detach().cpu().numpy()

    @staticmethod
    def getDefaultModel(n_epoch=2000, batch_size=256, label_batch_size=50,
                        consistency_scale=0.9, lr=0.05):
        # 控制无监督数据的损失权重， epoch达到consistency_rampup时候到达一
        consistency_rampup = int(n_epoch * 0.8)
        # consistency_scale = 0.
        # 无监督损失的基础模型
        baseModel = BaseCoTrain.getDefaultModel()
        model = SemiMLP(n_epoch, lr, consistency_scale, consistency_rampup, baseModel, None,
                        batch_size=batch_size, label_batch_size=label_batch_size)
        return model

    @staticmethod
    def getDefaultAdjustModel(n_epoch=1000, batch_size=256, label_batch_size=50,
                              consistency_scale=50, lr=0.05, ablation=False, addGaussianNoise=True, noise_scale=0.001):
        # 控制无监督数据的损失权重， epoch达到consistency_rampup时候到达一
        consistency_rampup = int(n_epoch * 0.8)
        # 无监督损失的基础模型
        baseModel = BaseCoTrain.getDefaultModel()
        model = SemiMLP(n_epoch, lr, consistency_scale, consistency_rampup, baseModel, None,
                        batch_size=batch_size, label_batch_size=label_batch_size, ablation=ablation,
                        addGaussianNoise=addGaussianNoise, noise_scale=noise_scale)
        return model

