import os

import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from sklearn.preprocessing import MinMaxScaler

from dataset.DatasetPreHandle import SimpleDataset
from util.ArffReaderUtil import read_data_arff


class ArffRegDataset(Dataset):

    def __init__(self, filename) -> None:
        super().__init__()
        self.filename = filename
        rows = read_data_arff(filename)
        self.X = rows[:, :-1]
        self.y = rows[:, -1]

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]

    def toSimpleDataset(self):
        return SimpleDataset(self.X, self.y)

    def train_test_split(self, train_size=0.8, isShuffle=True, random_state=None):
        # 划分测试集与训练集
        # 留出法
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=train_size, shuffle=isShuffle,
                                                            random_state=random_state)
        return SimpleDataset(X_train, y_train), SimpleDataset(X_test, y_test)

    def SSL_split(self, train_size=0.8, n_label=0.1, isShuffle=True, random_state=None):
        # 划分测试集与训练集
        # 留出法
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=train_size, shuffle=isShuffle
                                                            , random_state=random_state)
        trainData = SimpleDataset(X_train, y_train)
        testData = SimpleDataset(X_test, y_test)
        labelTrainData, unlabelTrainData = trainData.label_unlabel_split(n_label)
        return labelTrainData, unlabelTrainData, testData

    # add at 20221102
    def total_split(self, train_size=2000, n_label=0.2, eval_size=50, isShuffle=True, random_state=None):
        # 划分测试集与训练集
        X_train_eval, X_test, y_train_eval, y_test = train_test_split(self.X, self.y, train_size=train_size + eval_size,
                                                                      shuffle=isShuffle, random_state=random_state)
        testData = SimpleDataset(X_test, y_test)
        # 再次划分训练集与验证集
        X_eval, X_train, y_eval, y_train = train_test_split(X_train_eval, y_train_eval, train_size=eval_size,
                                                            shuffle=isShuffle, random_state=random_state)
        evalData = SimpleDataset(X_eval, y_eval)
        trainData = SimpleDataset(X_train, y_train)
        labelTrainData, unlabelTrainData = trainData.label_unlabel_split(n_label)
        return labelTrainData, unlabelTrainData, testData, evalData


    def transform(self, basePoints, D=None, isScaler=True):
        # 相对于基准点进行特征转化
        # newFeatureX = np.zeros([self.X.shape[0], basePoints.shape[0]])
        # for i, x in enumerate(self.X):
        #     for j, basePoint in enumerate(basePoints):
        #         distance = D(x, basePoint)
        #         newFeatureX[i, j] = distance
        if D is None:
            D = euclidean_distances
        newFeatureX = D(self.X, basePoints)
        # Notice： 这样转化过后的特征是没有归一化的
        # newFeatureX = np.sqrt(np.dot(self.X, basePoints))
        if isScaler:
            scaler = MinMaxScaler()
            self.X = scaler.fit_transform(newFeatureX)
        else:
            self.X = newFeatureX

    @staticmethod
    def getArffDatasetDict(path):
        datasetDict = {}
        filenames = os.listdir(path)
        for fn in filenames:
            absFilename = os.path.join(path, fn)
            if os.path.isfile(absFilename):
                dataset = ArffRegDataset(absFilename)
                key = fn.split(".")[0]
                datasetDict[key] = dataset
        return datasetDict

    @staticmethod
    def getAbaloneDataset(train_size=2000, n_label=0.025, random_state=1):
        dataPath = r"../data/abalone.arff"
        dataPath = os.path.join(os.path.dirname(__file__), dataPath)
        abalone = ArffRegDataset(dataPath)
        return abalone.SSL_split(train_size=train_size, n_label=n_label, random_state=random_state)

    @staticmethod
    def abalone():
        dataPath = r"../data/abalone.arff"
        dataPath = os.path.join(os.path.dirname(__file__), dataPath)
        abalone = ArffRegDataset(dataPath)
        return abalone


def main():
    path = "../data"
    datasetDict = ArffRegDataset.getArffDatasetDict(path)
    for (key, data) in datasetDict.items():
        print("dataset: {} n_size: {} n_feature: {} ".format(key, data.X.shape[0], data.X.shape[1]))


if __name__ == '__main__':
    main()
