import logging
import math

import numpy as np
from abc import ABC
from random import Random

import pandas as pd

from classifier import Classifier


class LogisticRegression(Classifier, ABC):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        super().__init__()
        self.w = None

    def calc_out(self, x):
        return 1/(1 + np.exp(- np.matmul(self.w, x.T)))

    def train(self, x_train, y_train):
        x_train = np.matrix(x_train)
        y_train = np.array(y_train)
        it = 1
        length = x_train.shape[0] - 1
        bias = np.ones((length + 1, 1))
        x_train = np.hstack((x_train, bias))
        self.w = np.random.rand(x_train.shape[1])
        error = np.asarray([1])
        while it < 100 or math.fabs(error.mean()) > 0.01:
            rd = Random().randint(0, length)
            x = x_train[rd]
            alfa = 1 / it
            out = self.calc_out(x)
            erro = y_train[rd] - out
            delta = (alfa * erro) * x
            self.w = self.w + delta
            error = np.append(error, [erro])
            it += 1
        pass

    def test(self, x_test, y_test):
        x_test = np.matrix(x_test)
        y_test = np.asarray(y_test)
        length = x_test.shape[0] - 1
        bias = np.ones((length + 1, 1))
        x_test = np.hstack((x_test, bias))
        y_pred = [1 if self.calc_out(x) > 0.5 else 0 for x in x_test]
        y_pred = np.asarray(y_pred)

        self.logger.warning("------- RESULTS------")

        self.logger.info(f"y_test: {y_test.T[0]}")
        self.logger.info(f"y_pred: {y_pred}")

        matrix = [[0, 0],
                  [0, 0]]

        for pos, ele in enumerate(y_test):
            matrix[int(ele)][int(y_pred[pos])] += 1

        self.logger.info("confusion matrix:")
        self.logger.info(f"\n{pd.DataFrame(matrix)}")
        error = [math.fabs(y_pred[i] - y_test[i]) for i in range(len(y_pred))]
        self.logger.info("final result:")
        self.logger.info(f"{(sum(error) / len(error)) * 100} % of error")

        return y_pred
