# preprocess.py

import pandas as pd
from packaging import version
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer




def load_and_clean(path):
    df = pd.read_csv(path)
    df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
    df['Person_Age'] = pd.to_numeric(df['Person_Age'], errors='coerce')
    df['Menarch_Age'] = pd.to_numeric(df['Menarch_Age'], errors='coerce')
    df['Menopause_Age'] = pd.to_numeric(df['Menopause_Age'], errors='coerce')
    df['Appointment_Date'] = pd.to_datetime(df['Appointment_Date'], errors='coerce')
    df['Month'] = df['Appointment_Date'].dt.month.fillna(0).astype(int)
    df['Weekday'] = df['Appointment_Date'].dt.weekday.fillna(0).astype(int)
    for col in ['Used_Contraceptives', 'Chronic_Diseases']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df.dropna(subset=['BMI', 'Person_Age', 'Menarch_Age'], inplace=True)
    df['Risk_Score'] = df[['Used_Contraceptives', 'Chronic_Diseases']].sum(axis=1)
    return df

def build_preprocessor(num_feats, cat_feats):
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        cat_pipe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        cat_pipe = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_feats),
        ('cat', cat_pipe, cat_feats)
    ])
    return preprocessor
