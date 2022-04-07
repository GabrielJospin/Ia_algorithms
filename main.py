import os
from datetime import datetime

import classifier
import connection
import logging
import json
from sklearn.model_selection import train_test_split


jFile = json.load(open("./config.json"))

PATH = jFile["path"]
SEP = jFile["separator"]
numOut = jFile["numOut"]
cls = jFile["classifier"]

now = str(datetime.now().isoformat())

if not os.path.exists(f"./log/{cls}/"):
    os.makedirs(f"./log/{cls}/")

logging.basicConfig(filename=f'./log/{cls}/{now}.log', encoding='UTF-8')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def getCls(cls):
    if cls.__eq__("DecisionTree"):
        logger.info("using Decision Tree classifier")
        return classifier.DecisionTree
    if cls.__eq__("LinearRegression"):
        logger.info("using Decision Linear Regression")
        return classifier.LinearRegression
    if cls.__eq__("LogisticRegression"):
        logger.info("using Decision Logistic Regression")
        return classifier.LogisticRegression
    else:
        logger.error("non-existent classifier")
        return classifier.Classifier


def main(cls=cls):
    cls = getCls(cls)

    conn = connection.Connection(PATH, SEP)
    df = conn.df

    X = df.iloc[:, 0:df.shape[1] - numOut]
    Y = df.iloc[:, df.shape[1] - numOut: df.shape[1]]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

    clasf = cls()
    logger.info(f"training with the dataset: {jFile['path']}")
    clasf.train(X_train, y_train)
    logger.info('trained')
    clasf.test(X_test, y_test)

    print("result in log files")

if __name__ == '__main__':
    main()
