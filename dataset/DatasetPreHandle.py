import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset


def numericByOnehot(nominalVal):
    reVal = None
    for i in range(nominalVal.shape[1]):
        # 确定种类的大小
        item = nominalVal[:, i]
        kindSet = set(item)
        n_kind = len(kindSet)
        # Fix set无序导致实验结果不可重复
        kindList = list(kindSet)
        # 排序， 使得结果稳定
        kindList.sort()
        itemHal = np.zeros(item.shape[0], dtype=int)
        for idx, kind in enumerate(kindList):
            itemHal[item == kind] = idx
        itemOneHot = np.eye(n_kind)[itemHal]
        if reVal is None:
            reVal = itemOneHot
        else:
            reVal = np.hstack((reVal, itemOneHot))
    return reVal


def numericByLabeLEncoder(nominalVal):
    reVal = None
    for i in range(nominalVal.shape[1]):
        # 确定种类的大小
        item = nominalVal[:, i]
        kindSet = set(item)
        itemHal = np.zeros(item.shape[0], dtype=int)
        for idx, kind in enumerate(kindSet):
            itemHal[item == kind] = idx
        itemLabelEn = itemHal.astype(float).reshape([-1, 1])
        if reVal is None:
            reVal = itemLabelEn
        else:
            reVal = np.hstack((reVal, itemLabelEn))
    return reVal


def normalize(dataMat):
    means = np.mean(dataMat, axis=0, keepdims=True)
    stds = np.std(dataMat, axis=0, keepdims=True)
    return (dataMat - means) / stds


def normalize2(dataMat, MAX=None, MIN=None):
    if MAX is None:
        MAX = np.max(dataMat, axis=0, keepdims=True)
    if MIN is None:
        MIN = np.min(dataMat, axis=0, keepdims=True)
    return (dataMat - MIN) / (MAX - MIN)


class RegressionDataset(Dataset):
    def __init__(self, file_path, nominalIdx) -> None:
        super().__init__()
        self.file_path = file_path
        data, meta = arff.loadarff(self.file_path)
        dfVal = pd.DataFrame(data).values
        # 分离离散属性
        nominalVal = dfVal[:, nominalIdx]
        # 分离连续属性
        numericIdx = [i for i in range(dfVal.shape[1]) if i not in nominalIdx]
        numericVal = dfVal[:, np.array(numericIdx)]
        # 离散属性：连续化操作
        nominalHalVal = numericByOnehot(nominalVal)
        # nominalHalVal = numericByLabeLEncoder(nominalVal)
        # 连续属性标准化
        # numericHalVal = normalize(numericVal.astype(float))
        numericHalVal = normalize2(numericVal.astype(float))
        # 拼接两个属性
        self.dataHal = np.hstack((nominalHalVal, numericHalVal))
        self.X = self.dataHal[:, :-1]
        self.y = self.dataHal[:, -1]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]

    def saveCSV(self, filename):
        np.savetxt(filename, self.dataHal, delimiter=',', fmt='%.6f')

    def train_test_split(self, mode=0, test_size=0.2, k=10):
        # 划分测试集与训练集
        if mode == 0:
            # 留出法
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=False)
            return SimpleDataset(X_train, y_train), SimpleDataset(X_test, y_test)
        else:
            # 交叉验证法
            datasetList = []
            kf = KFold(n_splits=k, shuffle=False)
            for train_index, test_index in kf.split(self.X):
                # print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                datasetList.append((SimpleDataset(X_train, y_train), SimpleDataset(X_test, y_test)))
            return datasetList


class SimpleDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]

    # 纵向分割数据集
    # TODO SHUFFLE
    def vertical_split(self, n_split=3, coverRatio=0):
        length = self.X.shape[1]
        lastIdx = 0
        datasetList = []
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
                dataX = np.hstack((self.X[:, startIdx:length], self.X[:, 0:endIdx % length]))
            else:
                dataX = self.X[:, startIdx:endIdx]
            datasetList.append(SimpleDataset(dataX, self.y))
        return datasetList

    def addPseudoLabel(self, x, y_halt):
        self.X = np.vstack((self.X, x))
        self.y = np.hstack((self.y, y_halt))

    @staticmethod
    def getPseudoLabelDataset(x, y_halt, dataset):
        X_new = np.vstack((dataset.X, x))
        y_new = np.hstack((dataset.y, y_halt))
        return SimpleDataset(X_new, y_new)

    def copy(self):
        return SimpleDataset(self.X.copy(), self.y.copy())

    def train_eval_split(self, train_size=0.9, isShuffle=True):
        # 划分测试集与训练集
        # 留出法
        X_train, X_eval, y_train, y_eval = train_test_split(self.X, self.y, train_size=train_size, shuffle=isShuffle)
        return SimpleDataset(X_train, y_train), SimpleDataset(X_eval, y_eval)

    def label_unlabel_split(self, n_label=0.1):
        # 对训练集另外划分有标签，与无标签
        n_label = int(n_label) if n_label > 1 else int(n_label * len(self.X))
        # 选择标签样本策略
        # 1. 选择前面的n_label个
        labelTrainX, labelTrainY = self.X[: n_label, ], self.y[: n_label]
        labelTrainDataset = SimpleDataset(labelTrainX, labelTrainY)

        unlabelTrainX, unlabelTrainY = self.X[n_label:, ], self.y[n_label:]
        unlabelTrainDataset = SimpleDataset(unlabelTrainX, unlabelTrainY)
        return labelTrainDataset, unlabelTrainDataset

    def setUnlabelInstance(self, unLabelIdxs, No_Label=-1):
        self.y[unLabelIdxs] = No_Label

    def setY(self, y):
        self.y = y


def main():
    path = "../data1/abalone.arff"
    nominalIdx = np.array([0])
    abaloneData = RegressionDataset(path, nominalIdx)
    abaloneData.saveCSV("abalone.csv")


if __name__ == '__main__':
    # setup_seed(10)
    main()
