import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader

from dataset.ArffRegDataset import ArffRegDataset
from dataset.DatasetPreHandle import SimpleDataset
from util import RegUtil
from util.RegUtil import train, trainAndEvaluate


class MLPDropoutReg(torch.nn.Module):
    def __init__(self, n_feature, n_output, regNNSize, p) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, regNNSize[0])
        self.dropout1 = torch.nn.Dropout(p=p)
        self.hidden1 = torch.nn.Linear(regNNSize[0], regNNSize[1])
        self.hidden2 = torch.nn.Linear(regNNSize[1], regNNSize[2])
        self.predict = torch.nn.Linear(regNNSize[2], n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.predict(x)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def getDefaultModel(n_input, n_output, p=0.2):
        hiddenLayers = [32, 256, 32]
        model = MLPDropoutReg(n_input, n_output, hiddenLayers, p)
        return model


# SkLearn 风格的方法
class MLP:

    def __init__(self, n_input, verbose=False) -> None:
        super().__init__()
        self.model = MLPDropoutReg.getDefaultModel(n_input, 1)
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        n_epoch = 500
        batch_size = 128
        isShuffle = True
        model = self.model

        model.to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        lossFunc = nn.MSELoss()
        dataset = SimpleDataset(X, y)
        dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=isShuffle)
        train(n_epoch, model, dataLoader, optimizer, lossFunc, self.device, self.verbose)

    def predict(self, X):
        # batch_size = 128
        # isShuffle = True
        # model = self.model
        # model.eval()
        # y = np.zeros([X.shape[0]])
        # dataset = SimpleDataset(X, y)
        # dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=isShuffle)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # predictY = None
        # with torch.no_grad():
        #     for idx, (data, label) in enumerate(dataLoader):
        #         data = data.to(device)
        #         predict = model(data)
        #         if predictY is None:
        #             predictY = predict
        #         else:
        #             predictY = torch.vstack((predictY, predict))
        # return predictY
        self.model.eval()
        with torch.no_grad():
            if not torch.is_tensor(X):
                X = torch.Tensor(X).to(self.device)
            return self.model(X).detach().cpu()


def main():
    torch.set_default_tensor_type(torch.DoubleTensor)
    path = "../data/abalone.arff"
    n_epoch = 500
    batch_size = 256
    isShuffle = True
    abaloneData = ArffRegDataset(path)
    # trainDataset, testDataset = abaloneData.train_test_split(train_size=0.1)
    trainDataset, unlabelTrainData, testDataset = abaloneData.SSL_split(train_size=2000, n_label=50, random_state=1)
    trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=isShuffle)
    testDataLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=isShuffle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = RegModel(8, 1, [32, 64, 32])
    n_input = abaloneData[0][0].shape[0]
    n_output = 1
    model = MLPDropoutReg.getDefaultModel(n_input, n_output, p=0.2)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.04, momentum=0.9)
    lossFunc = nn.MSELoss()
    trainAndEvaluate(n_epoch, model, trainDataLoader, testDataLoader, optimizer, lossFunc, device)


if __name__ == '__main__':
    RegUtil.setup_seed(10)
    #  n_epoch = 10000 p=0 0.089989886
    #  n_epoch = 10000 p=0.1 0.081030751
    #  n_epoch = 10000 p=0.2  0.078938712
    #  n_epoch = 10000 p=0.3 0.080692337
    #  n_epoch = 10000 p=0.5 0.086702911
    main()
