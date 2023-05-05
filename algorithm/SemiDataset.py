import numpy as np
import torch
from torch.utils.data import Dataset


class SemiDataset(Dataset):
    def __init__(self, X_l, y_l, X_u, NoLabel=-1, addGaussianNoise=True, noise_scale=0.001) -> None:
        super().__init__()
        self.X = np.vstack((X_l, X_u))
        idxArray = np.arange(self.X.shape[0])
        self.labelIdx = idxArray[: X_l.shape[0]]
        self.unlabelIdx = idxArray[X_l.shape[0]:]
        self.y = np.zeros(self.X.shape[0])
        self.y[self.labelIdx] = y_l
        self.y[self.unlabelIdx] = NoLabel
        self.addGaussianNoise = addGaussianNoise
        self.noise_scale = noise_scale

    def __getitem__(self, index):
        # return self.X[index], self.y[index]
        # add noise
        if self.addGaussianNoise:
            return self.X[index] + self.noise_scale * np.random.randn(*self.X[index].shape), self.y[index]
        else:
            return self.X[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]

    def getSemiIndex(self):
        return self.labelIdx, self.unlabelIdx
