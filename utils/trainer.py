from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return GaussianNB().fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)