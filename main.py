from CoTrain.BaseCoTrain import BaseCoTrain
from algorithm.SemiMLP import SemiMLP
from dataset.ArffRegDataset import ArffRegDataset
from util import RegUtil


def main():
    # torch.set_default_tensor_type(torch.DoubleTensor)
    n_epoch = 1000
    batch_size = 256
    label_batch_size = 50
    # 控制无监督数据的损失权重， epoch达到consistency_rampup时候到达一
    consistency_rampup = int(n_epoch * 0.8)
    # consistency_rampup = int(n_epoch * 1)
    # 控制无监督数据的损失权重最大为consistency_scale
    # consistency_scale = 0.9
    # consistency_scale = 0.
    consistency_scale = 50
    lr = 0.05
    # lr = 0.03
    # lr = 1e-3
    verbose = True
    abalone = ArffRegDataset.abalone()
    labelData, unlabelData, testData, evalData = abalone.total_split(n_label=0.1, random_state=1)
    # coReg = BaseCoTrain.getDefaultModel()
    # coReg.fit(labelData.X, labelData.y, unlabelData.X)
    # saferReg = SaferReg.getSaferReg()
    # baseModel = SaferReg.getKNN(3)
    # baseModel = MSSRA.getDefaultModel()
    baseModel = BaseCoTrain.getDefaultModel()
    # baseModel.fit(labelData.X, labelData.y)
    baseModel.fit(labelData.X, labelData.y, unlabelData.X)
    # model = SemiMLP(n_epoch, lr, consistency_scale, consistency_rampup, coReg, evalData,
    #                 batch_size=batch_size, label_batch_size=label_batch_size)
    model = SemiMLP(n_epoch, lr, consistency_scale, consistency_rampup, baseModel, evalData,
                    batch_size=batch_size, label_batch_size=label_batch_size, verbose=verbose)
    model.fit(labelData.X, labelData.y, unlabelData.X)
    rmse = RegUtil.evalRegTestRmse(model, testData)
    # rmse = RegUtil.evalRegTestRmse(model, testData)
    print("SemiMLP rmse: {}".format(rmse))
    # print("BaseCoTrain rmse: {}".format(RegUtil.evalRegTestRmse(coReg, testData)))
    print("baseModel rmse: {}".format(RegUtil.evalRegTestRmse(baseModel, testData)))


if __name__ == '__main__':
    RegUtil.setup_seed(10)
    main()
