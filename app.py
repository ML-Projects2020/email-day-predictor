# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import utility as util
from datetime import date
import locale
import os
# # Load the Linear Regression model
model = pickle.load(open(os.path.dirname(__file__)+"/email-day-prediction.pkl", 'rb'))

app = Flask(__name__)

def calculateDays(year, month, day):
    f_date = date(2020, 3, 1)
    l_date = date(year, month, day) #Selected date
    delta = l_date - f_date
    return np.array(delta.days+1)

@app.route('/')
def home():
    countries = util.getCountries()
    return render_template('index.html', countries=countries)
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        contactId = str(request.form['contactId'])
        campaignId = str(request.form['campaignId'])
        campaignDate = str(request.form['campaignDate'])
        country = str(request.form['country'])
        print("###############################")
        print(contactId+" "+campaignId+' '+campaignDate+" "+country)
        countries = util.getCountries()
        data = {'Contact ID': contactId,
                'campaign_id': campaignId,
                'Campaign LaunchTime': campaignDate,
                'Email Opened': 1
                }
        df = pd.DataFrame(data,index = [0])
        df['Campaign LaunchTime'] = pd.to_datetime(df['Campaign LaunchTime'])
        df['Campaign Day'] = df['Campaign LaunchTime'].dt.day_name()
        df['Cam Hour'] = df['Campaign LaunchTime'].dt.hour
        df['Cam Minute'] = df['Campaign LaunchTime'].dt.minute
        days_dict = {'Monday':1, 'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
        df['Campaign Day'] = df['Campaign Day'].map(days_dict)
        for index, value in enumerate(countries, start=1):
            if( country == value):
                print("&&&&&&&&&")
                df['country_'+str(index)] = 1
            else:
                df['country_'+str(index)] = 0
        X = df.drop(['Campaign LaunchTime'], axis=1)
        print("XXXXXXXXXXXXXXXXXXXX")
        print(X)
        prediction =  model.predict(X)
        print("PREDICTION", prediction[0])

        key_list = list(days_dict.keys()) 
        val_list = list(days_dict.values()) 
        
        return render_template('result.html', day=key_list[val_list.index(prediction[0])])

if __name__ == '__main__':
	app.run(debug=True)
