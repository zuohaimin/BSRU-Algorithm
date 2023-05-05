import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch import nn


def train(n_epoch, model, dataLoader, optimizer, lossFunc, device, verbose=False):
    for epoch in range(n_epoch):
        if verbose:
            print('Epoch [{}/{}]\n'.format(epoch + 1, n_epoch))
        model.train()
        sum_loss = 0.0
        for batch_idx, (trains, labels) in enumerate(dataLoader):
            length = len(dataLoader)
            optimizer.zero_grad()
            # model.zero_grad()
            trains = trains.to(device)
            labels = labels.to(device)
            predicts = model(trains)
            loss = lossFunc(predicts.squeeze(dim=1), labels)
            loss.backward()
            optimizer.step()
            # Tensor.item() 类型转换，返回一个数
            sum_loss += loss.item()
            # 注意这里是以一个batch为一个单位
            if verbose:
                print("[epoch:%d, iter:%d] RMSE Loss: %.03f"
                      % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1)))


def trainAndEvaluate(n_epoch, model, trainDataloader, testDataloader, optimizer, lossFunc, device):
    trainRmseList = []
    trainBatchIdxList = []
    testRmseList = []
    testBatchIdxList = []
    for epoch in range(n_epoch):
        print('Epoch [{}/{}]\r\n'.format(epoch + 1, n_epoch))
        model.train()
        sum_loss = 0.0
        total = 0.0
        for batch_idx, (trains, labels) in enumerate(trainDataloader):
            length = len(trainDataloader)
            optimizer.zero_grad()
            # model.zero_grad()
            trains = trains.to(device)
            labels = labels.to(device)
            predicts = model(trains)
            loss = lossFunc(predicts.squeeze(dim=1), labels)
            loss.backward()
            optimizer.step()
            # Tensor.item() 类型转换，返回一个数
            sum_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            # 注意这里是以一个batch为一个单位
            totalBatchIdx = batch_idx + 1 + epoch * length
            rmse = math.sqrt(sum_loss / total)
            print("[epoch:%d, iter:%d] Loss: %.09f"
                  % (epoch + 1, totalBatchIdx, rmse))
            trainRmseList.append(rmse)
            trainBatchIdxList.append(totalBatchIdx)
            if totalBatchIdx % 100 == 0:
                # 在测试集上评估
                testRmse = evalTestLoss(model, testDataloader, device)
                testRmseList.append(testRmse)
                testBatchIdxList.append(totalBatchIdx)
            # 切换为训练模式
            model.train()
    # 模型训练结束，在训练集上评估模型质量
    evalTestLoss(model, testDataloader, device)
    plotRmseLoss(trainBatchIdxList, trainRmseList, testBatchIdxList, testRmseList)


def saveCheckPoint(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def evalTestLoss(net, testDataloader, device):
    net.eval()
    sumLoss = 0.0
    total = 0.0
    evallossFunc = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        for idx, (data, label) in enumerate(testDataloader):
            data = data.to(device)
            label = label.to(device)
            predict = net(data.to(device))
            loss = evallossFunc(predict.squeeze(dim=1), label)
            sumLoss += loss.item()
            total += label.size(0)
    meanLoss = sumLoss / total
    rmse = math.sqrt(meanLoss)
    print("Evaluate On Test Dataset, RMSE Loss: {:<.9f}".format(rmse))
    return rmse


def evalNNLoss(net, evalData, device):
    net.eval()
    sumLoss = 0.0
    total = 0.0
    evallossFunc = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        data, label = torch.Tensor(evalData.X), torch.Tensor(evalData.y)
        data = data.to(device)
        label = label.to(device)
        predict = net(data.to(device))
        loss = evallossFunc(predict.squeeze(dim=1), label)
        sumLoss += loss.item()
        total += label.size(0)
    meanLoss = sumLoss / total
    rmse = math.sqrt(meanLoss)
    print("Evaluate On eval Dataset, RMSE Loss: {:<.9f}".format(rmse))
    return rmse


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return time_dif


# https://blog.csdn.net/Zzz_zhongqing/article/details/107528717
# L2 正则化
def L2Loss(model, alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(parma, 2)))
    return l2_loss


def L1Loss(model, beta):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + beta * torch.sum(torch.abs(param))
    return l1_loss


def setup_seed(seed):
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def experiment(func, n_loop=100):
    # 计算平均运行时间
    # 计算输出的的均值 ± 标准差
    start = time.time()
    resultList = np.zeros(n_loop)
    for idx in range(n_loop):
        resultList[idx] = func()
    print('Running time: %s Seconds' % (get_time_dif(start) / n_loop))
    print("Experiment in {} time, {}±{}， min：{}".format(n_loop, resultList.mean(), resultList.std(), resultList.min()))


def experiments(func, funcNames, n_loop=100):
    # 计算平均运行时间
    # 计算输出的的均值 ± 标准差
    start = time.time()
    resultList = np.zeros([n_loop, len(funcNames)])
    for idx in range(n_loop):
        resultList[idx] = func()
    print("{} Running time: {} Seconds".format(str(func), get_time_dif(start) / n_loop))
    meanList = np.mean(resultList, axis=0)
    stdList = np.std(resultList, axis=0)
    minList = np.min(resultList, axis=0)
    for idy, name in enumerate(funcNames):
        print("{} Experiment in {} time, {}±{}， min：{}"
              .format(name, n_loop, meanList[idy], stdList[idy], minList[idy]))


def plotRmseLoss(X1, Y1, X2, Y2, title="RMSE LOSS On {}", comment="abalone"):
    # plt.title(title.format(comment))
    plt.figure(figsize=(6, 5))
    plt.xlabel('batchIdx')
    plt.ylabel('RMSE')
    line1, = plt.plot(X1, Y1, label="train Dataset")
    line2, = plt.plot(X2, Y2, label="eval Dataset")
    plt.legend(handles=[line1, line2], loc='best')
    plt.show()


# 横向分割
def vertical_split(X, n_split=3, coverRatio=0):
    length = X.shape[1]
    lastIdx = 0
    subXList = []
    for idx in range(n_split):
        if idx == 0:
            startIdx = 0
        else:
            startIdx = int(lastIdx - coverRatio * length)
        endIdx = int(startIdx + (1 / n_split + coverRatio) * length)
        if idx == n_split - 1 and endIdx < length:
            endIdx = length
        lastIdx = endIdx
        if endIdx > length:
            dataX = np.hstack((X[:, startIdx:length], X[:, 0:endIdx % length]))
        else:
            dataX = X[:, startIdx:endIdx]
        subXList.append(dataX)
    return subXList


def evalRegTestRmse(reg, testDataset):
    return mean_squared_error(testDataset.y, reg.predict(testDataset.X), squared=False)


def evalRegTestMse(reg, testDataset):
    return mean_squared_error(testDataset.y, reg.predict(testDataset.X), squared=True)


def evalRegTestR2(reg, testDataset):
    from sklearn.metrics import r2_score
    return r2_score(testDataset.y, reg.predict(testDataset.X))


def experimentAllAlgrithm(algrithmDict, trainData, testData, squared=False):
    lossList = []
    for idx, (name, regModel) in enumerate(algrithmDict.items()):
        regModel.fit(trainData.X, trainData.y, testData.X)
        rmse = mean_squared_error(testData.y, regModel.predict(testData.X), squared=squared)
        print("Algorithm: {} RMSE:{}".format(name, rmse))
        lossList.append(rmse)
    return lossList


def experimentAllSemiAlgrithm(algrithmDict, labelData, unlabelData, testData, evaluate_metric="RMSE"):
    from sklearn.metrics import mean_squared_error  # MSE
    from sklearn.metrics import mean_absolute_error  # MAE
    from sklearn.metrics import r2_score  # R 2
    lossList = []
    for idx, (name, regModel) in enumerate(algrithmDict.items()):
        regModel.fit(labelData.X, labelData.y, unlabelData.X)
        y_pre = regModel.predict(testData.X)
        metric = None
        if evaluate_metric == "RMSE":
            metric = mean_squared_error(testData.y, y_pre, squared=False)
        elif evaluate_metric == "MSE":
            metric = mean_squared_error(testData.y, y_pre, squared=True)
        elif evaluate_metric == "MAE":
            metric = mean_absolute_error(testData.y, y_pre)
        elif evaluate_metric == "R2":
            metric = r2_score(testData.y, y_pre)
        else:
            raise Exception("No Available Metric, evaluate_metric: {}".format(evaluate_metric))
        print("Algorithm: {} {}:{}".format(name, evaluate_metric, metric))
        lossList.append(metric)
    return lossList
