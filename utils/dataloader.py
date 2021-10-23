import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        
        # binning with cut
        self.dataset['age'] = pd.qcut(self.dataset['age'], 15)
        
        # filling bmi nans with normal dtribution m=70
        bmi_std = self.dataset['bmi'].std()
        bmi_nan_count = self.dataset['bmi'].isna().sum()
        rng = np.random.RandomState(42)
        self.dataset['bmi'] = self.dataset['bmi'].fillna(value=pd.Series(
            np.round(np.random.normal(70, bmi_std/2.0, bmi_nan_count),1), 
            index=self.dataset[np.isnan(self.dataset['bmi'])].index
            ))
        
        # normalization
        self.dataset['avg_glucose_level'] = MinMaxScaler().fit_transform(self.dataset['avg_glucose_level'].values.reshape(-1, 1))
        self.dataset['bmi'] = MinMaxScaler().fit_transform(self.dataset['bmi'].values.reshape(-1, 1))

        # drop columns
        drop_elements = ['id', 'gender', 'Residence_type']
        self.dataset = self.dataset.drop(drop_elements, axis=1)

        # encode labels
        le = LabelEncoder()
        
        le.fit(self.dataset['age'])
        self.dataset['age'] = le.transform(self.dataset['age'])

        le.fit(self.dataset['ever_married'])
        self.dataset['ever_married'] = le.transform(self.dataset['ever_married'])
        
        le.fit(self.dataset['work_type'])
        self.dataset['work_type'] = le.transform(self.dataset['work_type'])
        
        le.fit(self.dataset['smoking_status'])
        self.dataset['smoking_status'] = le.transform(self.dataset['smoking_status'])

        return self.dataset