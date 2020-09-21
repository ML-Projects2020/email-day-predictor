import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from datetime import date
import pickle
import io
import os
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import utility as encode

df = pd.read_csv(os.path.dirname(__file__)+'/Contactss.csv')

# Replacing Null Values (Hardcoding)
df['country'].fillna('missing',inplace=True)
df['state'].fillna('Empty',inplace=True)

df =  df.join(encode.country_encode(df))
# df =  df.join(encode.state_encode(df))

df.drop('country', axis=1, inplace=True)
df.drop('state', axis=1, inplace=True)

df['Email Opened Time'] = pd.to_datetime(df['Email Opened Time'])
df['Campaign LaunchTime'] = pd.to_datetime(df['Campaign LaunchTime'])

df['Day'] = df['Email Opened Time'].dt.day_name()
df.drop('Email Opened Time', axis=1, inplace=True)

# df['Hour'] = df['Email Opened Time'].dt.hour
# df['Minute'] = df['Email Opened Time'].dt.minute

df['Campaign Day'] = df['Campaign LaunchTime'].dt.day_name()
df['Cam Hour'] = df['Campaign LaunchTime'].dt.hour
df['Cam Minute'] = df['Campaign LaunchTime'].dt.minute

df['Day'] = df['Day'].map( {'Monday':1, 'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7} )
df['Campaign Day'] = df['Campaign Day'].map( {'Monday':1, 'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7} )

X = df.drop(['Day','parent_campaign_id','Campaign LaunchTime'], axis=1)

Y=df['Day']

# rus = RandomOverSampler(random_state=40)
# X_res, y_res =  rus.fit_resample(X, Y)  

X_train,X_test,y_train,y_test=train_test_split(X, Y, test_size=0.3, 
                                                    random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train,y_train)

filename = os.path.dirname(__file__)+'/email-day-prediction.pkl'
pickle.dump(rf, open(filename, 'wb')) 

y_pred=rf.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))