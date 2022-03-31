import classifier
import connection
import logging
import json
from sklearn.model_selection import train_test_split

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

jFile = json.load(open("./config.json"))

PATH = jFile["path"]
numOut = jFile["numOut"]
cls = jFile["classifier"]


def getCls(cls):
    if cls.__eq__("DecisionTree"):
        logger.info("using Decision Tree classifier")
        return classifier.DecisionTree
    else:
        logger.error("non-existent classifier")
        return classifier.Classifier


def main(cls=cls):
    cls = getCls(cls)

    conn = connection.Connection(PATH)
    df = conn.df

    X = df.iloc[:, 0:df.shape[1] - numOut]
    Y = df.iloc[:, df.shape[1] - numOut: df.shape[1]]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

    clasf = cls()
    clasf.train(X_train, y_train)
    y_pred = clasf.test(X_test)

    print(y_pred)
    print(y_test)

if __name__ == '__main__':
    main()
