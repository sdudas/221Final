import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PreProcess:
    def __init__(self, file):
        self.path = file
        self.sensitive_variables = []
        self.irrelevant_variables = ["id", "name", "first", "last", "compas_screening_date", "dob", "age", "days_b_screening_arrest", "c_charge_desc", "c_jail_in", 
        "c_jail_out", "c_case_number", "c_offense_date", "c_arrest_date", "c_days_from_compas", "r_case_number", "r_days_from_arrest", "r_offense_date", "r_charge_desc", "r_jail_in", "r_jail_out", "vr_case_number", "vr_charge_degree", "vr_offense_date", "vr_charge_desc", "type_of_assessment", "screening_date", "v_screening_date","v_type_of_assessment", "in_custody", "out_custody", "start", "end", "event"]
        self.to_predict = ['score_text', 'score_text_cat']

    def read_in_data(self):
        input_data = pd.read_csv(self.path)
        df = pd.DataFrame(input_data)
        df = df[(df.days_b_screening_arrest <= 30)
                & (df.days_b_screening_arrest >= -30)
                & (df.is_recid != -1)
                & (df.c_charge_degree != 'O')
                & (df.score_text != 'N/A')]
        df = df.drop(columns=(self.sensitive_variables + self.irrelevant_variables))

        obj_df = df.select_dtypes(include=['object']).copy
        df["score_text"] = df["score_text"].astype('category')
        df["score_text_cat"] = df["score_text"].cat.codes

        y = (df[["score_text_cat"]])
        df = df.drop(columns=self.to_predict)
        df = df.fillna('Unknown').pipe(pd.get_dummies, drop_first=True)
        categorical_df = df.select_dtypes(include=['object']).copy().columns.tolist()
        X = pd.get_dummies(df, columns=categorical_df)
        return X, y

        def split_train_test(self, X, y):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
            return X_train, X_test, y_train, y_test

pp = PreProcess("data/compas-scores-two-years.csv")
pp.read_in_data()