from sklearn.svm import SVC
from lightgbm import LGBMClassifier


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return LGBMClassifier().fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return (trained.predict_proba(test_x)[:,1] >= 0.386).astype(bool)