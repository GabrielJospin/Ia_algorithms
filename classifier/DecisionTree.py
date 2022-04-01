import math
import numpy as np
from abc import ABC
import classifier as cls
import utils.sheet


class DecisionTree(cls.Classifier, ABC):

    def __init__(self):
        super().__init__()

    def train(self, x_train, y_train):
        ent_tot = self.calc_amb_entropy(y_train)
        print(ent_tot)
        ent = self.matrix_entropy(x_train, y_train)
        prob = y_train[y_train.Vai_esperar == 0].shape[0]/y_train.shape[0]
        print(ent)
        ent_max = max(ent)
        pos_max = ent.index(ent_max)
        options = set(np.asarray(x_train.iloc[:, pos_max:pos_max + 1]).flatten())
        print(options)
        print(prob)
        tree = utils.sheet(pos_max, ent, prob)
        aux = tree
        x_list = x_train
        y_list = y_train
        while ent_max < ent_tot or tree.height() < len(ent):
            for i in options:
                aux.creat_add_son(ent_tot - ent_max,
                                  sum(y_list[x_train[pos_max] == i]) / len(y_list[x_train[pos_max] == i]),
                                  i)



        pass

    def test(self, x_test):
        pass

    @staticmethod
    def calc_binary_ent(prob):
        if prob == 0:
            return -(1 - prob) * math.log2(1 - prob)
        if prob == 1:
            return -(prob * math.log2(prob))

        return -(prob * math.log2(prob) + (1 - prob) * math.log2(1 - prob))

    def calc_amb_entropy(self, list_in):
        list_in = np.asarray(list_in).T
        prob = list_in[list_in == 1].shape[0] / len(list_in[0])
        ent = self.calc_binary_ent(prob)
        return ent

    def calc_gain(self, list_in, list_out):


        ent_geral = self.calc_amb_entropy(list_out)

        list_in = np.asarray(list_in.T).T[0]
        list_out = np.asarray(list_out).T[0]
        tam = len(list_in)

        options = set(list_in.flatten())
        summary = 0

        for op in options:
            exemp_list_in = list_in[list_in == op]
            exemp_list_out = list_out[list_in == op]
            prob_glob = len(exemp_list_in) / tam
            prob_loc = exemp_list_out.sum() / len(exemp_list_out)
            summary += prob_glob * self.calc_binary_ent(prob_loc)

        return ent_geral - summary

    def matrix_entropy(self, matrix, out):
        matrix = np.matrix(matrix)
        hent = list()
        for collumn in matrix.T:
            hent.append(self.calc_gain(collumn, out))

        return hent
