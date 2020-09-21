
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
def country_encode(df):
    oex = OneHotEncoder()
    enc_df = pd.DataFrame(oex.fit_transform(df[['country']]).toarray()).add_prefix('country_')
    return enc_df

def state_encode(df):
    oex = OneHotEncoder()
    enc_df = pd.DataFrame(oex.fit_transform(df[['state']]).toarray()).add_prefix('state_')
    return enc_df

def getCountries():
    df = pd.read_csv(os.path.dirname(__file__)+'/Contactss.csv')
    df['country'].fillna('missing',inplace=True)
    return df['country'].unique()
