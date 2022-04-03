import math
import os.path
import logging
import numpy as np
from abc import ABC
import pandas as pd
import classifier as cls
from utils.sheet import sheet
from datetime import datetime

now = str(datetime.now().isoformat())

if not os.path.exists("./log/DecisionTree/"):
    os.makedirs("./log/DecisionTree/")

logging.basicConfig(filename=f'./log/DecisionTree/{now}.log', encoding='UTF-8')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DecisionTree(cls.Classifier, ABC):

    def __init__(self):
        super().__init__()
        self.tree = None

    def train(self, x_train, y_train):
        self.tree = self.create_tree(x_train, y_train, sheet(None, None, None))
        pass

    def create_tree(self, x_train, y_train, tree: sheet):
        y_train = np.asarray(y_train)
        ent_tot = self.calc_amb_entropy(y_train)
        ent = self.matrix_entropy(x_train, y_train)
        ent_max = max(ent)
        pos_max = ent.index(ent_max)
        options = set(np.asarray(x_train.iloc[:, pos_max:pos_max + 1]).flatten())
        tree.entropy = ent_tot
        tree.probability = sum(y_train)/len(y_train)
        tree.column = pos_max

        if tree.probability >= 1 or tree.probability <= 0:
            return tree

        for op in options:
            fil = x_train.iloc[:, pos_max] == op
            if x_train.loc[fil].shape[0] > 1:
                tree.sons[op] = self.create_tree(x_train.loc[fil], y_train[fil], sheet(None, None, None))
        return tree

    def test(self, x_test, y_test):
        y_test = np.asarray(y_test.iloc[:]).reshape(1, -1)[0]
        y_pred = []
        for x in x_test.iterrows():
            y_pred.append(1 if self.run(x) > 0.5 else 0)

        logger.warning("------- RESULTS------")

        logger.info(f"y_test: {y_test}")
        logger.info(f"y_pred: {y_pred}")

        matrix = [[0, 0],
                  [0, 0]]

        for pos, ele in enumerate(y_test):
            matrix[int(ele)][int(y_pred[pos])] += 1

        logger.info("confusion matrix:")
        logger.info(f"\n{pd.DataFrame(matrix)}")
        error = [math.fabs(y_pred[i] - y_test[i]) for i in range(len(y_pred))]
        logger.info("final result:")
        logger.info(f"{1 - sum(error)/len(error)} % of error")

        return y_pred

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

    def run(self, x):
        tree = self.tree
        x = np.asarray(x[1])
        while len(tree.sons) > 0:
            if x[tree.column] not in tree.sons.keys():
                return 0.5
            tree = tree.sons[x[tree.column]]

        return tree.probability

