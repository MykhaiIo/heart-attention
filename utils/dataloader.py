import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        
        # replace
        self.dataset['gender'] = self.dataset['gender'].replace(['Other'], 'Male')
        
        # binning with cut
        self.dataset['age'] = pd.qcut(self.dataset['age'], 4)
        self.dataset['avg_glucose_level'] = pd.cut(self.dataset['avg_glucose_level'], 3)

        # drop columns
        drop_elements = ['id', 'gender', 'hypertension', 'Residence_type', 'bmi']
        self.dataset = self.dataset.drop(drop_elements, axis=1)

        # encode labels
        le = LabelEncoder()
        
        le.fit(self.dataset['age'])
        self.dataset['age'] = le.transform(self.dataset['age'])

        le.fit(self.dataset['heart_disease'])
        self.dataset['heart_disease'] = le.transform(self.dataset['heart_disease'])

        le.fit(self.dataset['ever_married'])
        self.dataset['ever_married'] = le.transform(self.dataset['ever_married'])
        
        le.fit(self.dataset['work_type'])
        self.dataset['work_type'] = le.transform(self.dataset['work_type'])
        
        le.fit(self.dataset['avg_glucose_level'])
        self.dataset['avg_glucose_level'] = le.transform(self.dataset['avg_glucose_level'])
        
        le.fit(self.dataset['smoking_status'])
        self.dataset['smoking_status'] = le.transform(self.dataset['smoking_status'])

        return self.dataset