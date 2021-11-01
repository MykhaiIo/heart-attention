import pickle

import sys
sys.path.append("./")

from settings.constants import SAVED_ESTIMATOR


class Predictor:
    def __init__(self):
        self.loaded_estimator = pickle.load(open(SAVED_ESTIMATOR, 'rb'))

    def predict(self, data):
        return (self.loaded_estimator.predict_proba(data)[:,1] >= 0.386).astype(bool)