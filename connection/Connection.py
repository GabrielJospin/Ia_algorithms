import pandas
import pandas as pd


class Connection:

    def __init__(self, path, sep=';'):
        self.path = path
        self.df = pandas.read_csv(path, delimiter=sep)
