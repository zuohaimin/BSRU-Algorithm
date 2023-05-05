from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

from algorithm.SklearnInterface import SklearnInterface
from dataset.ArffRegDataset import ArffRegDataset
from dataset.DatasetPreHandle import SimpleDataset
from util import RegUtil


class BaseCoTrain(SklearnInterface, metaclass=ABCMeta):
    """
        定义协同训练细节的接口类
    """

    def __init__(self, T, configs, poolSize) -> None:
        super().__init__()
        self.T = T
        self.configs = configs
        self.poolSize = poolSize
        self.pi = [None, None]
        # 带标签样本
        self.L = []
        # 模型
        self.models = []
        self.U = None

    def fit(self, X_l, y_l, X_u):
        # 初始化有状态的变量
        self.pi = [None, None]
        self.L = []
        self.models = []
        self.U = X_u.copy()
        for i, config in enumerate(self.configs):
            if config["metric"] == "mahalanobis":
                X = np.vstack((X_l, self.U))
                config["metric_params"] = {"VI": np.linalg.inv(np.cov(X.T)).T}
            self.L.append(SimpleDataset(X_l.copy(), y_l.copy()))
            self.models.append(self.kNN(self.L[i], config))
        U_t = self.get_unlabel_pool(self.U)
        # T次
        for t in range(self.T):
            # 两个模型
            # 选择样本的数量
            replenish_count = 0
            self.pi = [None, None]
            for j, model in enumerate(self.models):
                y_halts = model.predict(U_t)
                Omega_us = self.Neighbors(U_t, self.L[j], self.configs[j])
                confidence_scores = np.zeros(U_t.shape[0])
                for idx, x_u in enumerate(U_t):
                    y_halt = y_halts[idx]
                    Omega_u = Omega_us[idx]
                    # 构造添加后的标记数据
                    newL = self.L[j].copy()
                    newL.addPseudoLabel(x_u, y_halt)
                    newModel = self.kNN(newL, self.configs[j])
                    confidence_score = self.cal_confidence_score(model, newModel, self.L[j], Omega_u)
                    confidence_scores[idx] = confidence_score
                if confidence_scores.max() > 0:
                    chooseIndex = np.argmax(confidence_scores)
                    self.pi[1 - j] = (U_t[chooseIndex], y_halts[chooseIndex])
                    # 将选择样本从缓冲池中删除
                    U_t = np.delete(U_t, chooseIndex, 0)
                    replenish_count += 1
            # 没有样本可以选择，就直接结束循环
            if replenish_count == 0:
                print("replenish_count == 0 break!")
                break
            # 添加伪标签
            for L_i, pii, model in zip(self.L, self.pi, self.models):
                if pii is None:
                    continue
                x_u, y_halt = pii
                L_i.addPseudoLabel(x_u, y_halt)
                # 在新数据上重新拟合模型
                model.fit(L_i.X, L_i.y)
            # 更新缓冲池U_t, 无标记数据self.U
            updateIdx = np.random.choice(self.U.shape[0], replenish_count, replace=False)
            U_t = np.vstack((U_t, self.U[updateIdx]))
            self.U = np.delete(self.U, updateIdx, 0)
            assert U_t.shape[0] == self.poolSize

    def predict(self, X):
        predictY = None
        for model in self.models:
            y_pre = model.predict(X)
            if predictY is None:
                predictY = y_pre
            else:
                predictY = np.vstack((predictY, y_pre))
        meanPreY = np.mean(predictY, axis=0)
        return meanPreY

    def get_unlabel_pool(self, X_u):
        U = X_u.copy()
        chooseIdx = np.random.choice(U.shape[0], self.poolSize, replace=False)
        U_t = U[chooseIdx]
        return U_t

    @staticmethod
    def Neighbors(x_u, L_j, config_j):
        nn = NearestNeighbors()
        nn.set_params(**config_j)
        nn.fit(L_j.X)
        _, indices = nn.kneighbors(x_u)
        return indices

    @staticmethod
    def kNN(L, config):
        # init and fit
        knn = KNeighborsRegressor()
        knn.set_params(**config)
        knn.fit(L.X, L.y)
        return knn

    @staticmethod
    def cal_confidence_score(beforeModel, afterModel, L, Omega_u):
        """
        计算为伪标签置信度
        :param afterModel:
        :param beforeModel:
        :param L:
        :param Omega_u:
        :return:
        """
        # for idx in Omega_u:
        #     l_X, l_y = L[idx]
        #     l_X = l_X.reshape([1, -1])
        #     delta_x_u += ((l_y - knnReg.predict(l_X)) ** 2 - (l_y - newKnnReg.predict(l_X)) ** 2)
        L_x, L_y = L[Omega_u]
        predictBefore = beforeModel.predict(L_x)
        predictAfter = afterModel.predict(L_x)
        delta_x_u = (L_y - predictBefore) ** 2 - (L_y - predictAfter) ** 2
        confidence_score = delta_x_u.mean()
        return confidence_score

    @staticmethod
    def getDefaultModel():
        n_round = 100
        poolSize = 100
        config1 = {"n_neighbors": 3, "metric": "mahalanobis"}
        config2 = {"n_neighbors": 3, "metric": "euclidean"}
        # config2 = {"n_neighbors": 3, "metric": "cosine"}

        config = [config1, config2]
        model = BaseCoTrain(n_round, config, poolSize)
        return model


def main():
    baseCoTrain = BaseCoTrain.getDefaultModel()
    # path = "../../data/abalone.arff"
    # abanlone = ArffRegDataset(path)
    # trainData, testData = abanlone.train_test_split(100, random_state=1)
    labelTrainData, unlabelTrainData, testData = ArffRegDataset.getAbaloneDataset(random_state=1)
    for i in range(10):
        baseCoTrain.fit(labelTrainData.X, labelTrainData.y, unlabelTrainData.X)
        rmse = RegUtil.evalRegTestRmse(baseCoTrain, testData)
        # SSRCoreg Rmse: 0.11827706539314246
        # SSRCoreg Rmse: 0.09117208166552124
        # BaseCoTrain Rmse: 0.09162752531757794
        # BaseCoTrain Rmse: 0.095705904826906
        # "metric": "cosine" BaseCoTrain Rmse: 0.09496202694132222
        print("BaseCoTrain Rmse: {}".format(rmse))


if __name__ == '__main__':
    # SSRCoreg Rmse: 0.12444868930411361
    RegUtil.setup_seed(10)
    main()
