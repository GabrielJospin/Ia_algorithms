from abc import ABC, abstractmethod


class Classifier(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def test(self, x_test):
        pass
