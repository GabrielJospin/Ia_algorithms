import pandas
import pandas as pd


class Connection:

    def __init__(self, path):
        self.path = path
        self.df = pandas.read_csv(path, delimiter=";")
