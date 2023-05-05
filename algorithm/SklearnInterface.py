from abc import abstractmethod, ABCMeta


class SklearnInterface(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X_l, y_l, X_u):
        pass

    @abstractmethod
    def predict(self, X):
        pass

