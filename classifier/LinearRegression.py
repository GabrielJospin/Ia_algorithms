from abc import ABC
from classifier import Classifier


class LinearRegression(Classifier, ABC):

    def __init__(self):
        super().__init__()
        self.w = None

    def train(self, x_train, y_train):

        pass

    def test(self, x_test, y_test):
        pass