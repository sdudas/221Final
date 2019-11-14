import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
DataProcessor class contains the methods to load, preprocess and split the data into train and test set.
"""
class DataProcessor:

    def __init__(self, file):
        self.path = file

        self.sensitive_variables = []
        self.irrelevant_variables = ["id", "name", "first", "last", "compas_screening_date", "dob", "age", "days_b_screening_arrest", "c_charge_desc", "c_jail_in", 
        "c_jail_out", "c_case_number", "c_offense_date", "c_arrest_date", "c_days_from_compas", "r_case_number", "r_days_from_arrest", "r_offense_date", "r_charge_desc", "r_jail_in", "r_jail_out", "vr_case_number", "vr_charge_degree", "vr_offense_date", "vr_charge_desc", "type_of_assessment", "screening_date", "v_screening_date","v_type_of_assessment", "in_custody", "out_custody", "start", "end", "event"]
        self.to_predict = ['score_text', 'score_text_cat']

    """
    Pulling the data in raw format found here: 
    DataProcessing done as:
    1. Identifying the sensitive attribute of the data
    2. Dropping the sensitive attribute from the dataFrame
    """

    def loadData(self, sensitive_attribute, attribute, predictionValue, prediction_column):
        input_data = pd.read_csv(self.path)
        df = pd.DataFrame(input_data)
        """ 
            Perform the same preprocessing as the original analysis by Pro-Publica
            https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        """
        # df = df[(df.days_b_screening_arrest <= 30)
        #         & (df.days_b_screening_arrest >= -30)
        #         & (df.is_recid != -1)
        #         & (df.c_charge_degree != 'O')
        #         & (df.score_text != 'N/A')]

        input_data = pd.read_csv(self.path)
        df = pd.DataFrame(input_data)
        df = df[(df.days_b_screening_arrest <= 30)
                & (df.days_b_screening_arrest >= -30)
                & (df.is_recid != -1)
                & (df.c_charge_degree != 'O')
                & (df.score_text != 'N/A')]
        print(df.columns)


        obj_df = df.select_dtypes(include=['object']).copy
        df["score_text"] = df["score_text"].astype('category')
        df["score_text_cat"] = df["score_text"].cat.codes

        sensitive_attribs = [sensitive_attribute]
        Z = self.split_columns(df, sensitive_attribs, sensitive_attribute, attribute)

        y = (df[["score_text_cat"]])
        df = df.drop(columns=self.to_predict)
        df = df.fillna('Unknown').pipe(pd.get_dummies, drop_first=True)
        categorical_df = df.select_dtypes(include=['object']).copy().columns.tolist()
        X = pd.get_dummies(df, columns=categorical_df)
        return X, y, Z

    """
    Split the sensitive attribute column so that this is not part of training set
    """
    def split_columns(self, df, sensitive_attribs, sensitive_attribute, attribute):
        Z = (df.loc[:, sensitive_attribs].assign(
            new_column=lambda df: (df[sensitive_attribute] == attribute).astype(int)))
        Z.drop(columns=[sensitive_attribute], inplace=True)
        Z.rename(columns={'new_column': sensitive_attribute}, inplace=True)
        return Z


    """
    Split the data into train and test set.
    """
    def split_data(self, X, y, Z):
        X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, test_size=0.3, stratify=y)
        # standardize the data
        scaler = StandardScaler().fit(X_train)
        scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
        X_train = X_train.pipe(scale_df, scaler)
        X_test = X_test.pipe(scale_df, scaler)

        return X_train, X_test, y_train, y_test, Z_train, Z_test




