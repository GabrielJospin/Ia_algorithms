import math

import numpy as np
from abc import ABC

import classifier as cls


class DecisionTree(cls.Classifier, ABC):

    def __init__(self):
        super().__init__()

    def train(self, x_train, y_train):
        hent_tot = self.calc_entropy(y_train)
        print(hent_tot)
        hent = self.matrix_entropy(x_train)
        print(hent)
        hent_max = max(hent)
        pos_max = hent.index(hent_max)
        options = set(np.asarray(x_train.iloc[:, pos_max:pos_max+1]).ravel())
        print(options)
        pass

    def test(self, x_test):
        pass

    @staticmethod
    def calc_entropy(list_in):
        options = set(np.asarray(list_in).ravel())
        list_in = np.array(list_in).T
        x = len(options)
        sum = 0
        for i in options:
            prob = list_in[list_in == i].shape[0] / len(list_in)
            prob_in = list_in[list_in == i].sum() / list_in[list_in == i].shape[0]
            if prob_in == 0:
                sum += 0
            else:
                sum += prob * math.log2(prob_in)
        return sum

    def matrix_entropy(self, matrix):
        matrix = np.matrix(matrix)
        hent = list()
        for collumn in matrix.T:
            hent.append(self.calc_entropy(collumn))

        return hent

