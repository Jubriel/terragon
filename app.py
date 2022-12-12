import streamlit as st
import pandas as pd
import joblib

# Title
st.header("Terragon Click rate App")

c = st.container()
date = c.date_input("Select Date")
time = c.time_input("Select Time")

col1, col2 = st.columns(2,gap='medium')


col2.subheader('Customer Transactions')
# Input bar 1
total = col2.number_input("Enter Total Amount Spent")
# Input bar 2
voice = col2.number_input("Enter Amount Spent on Calls")
# Input bar 3
data = col2.number_input("Enter Amount Spent on Data")
# Input bar 4
sms = col2.number_input("Enter Total SMS Cost")
# Input bar 5
data_talk_rev = col2.number_input("Enter Data-Talk Revenue")

col1.subheader('Customer Demography')
age = col1.number_input("Enter Customer Age")
cus_cls = col1.number_input("Enter Customer Class")
# Dropdown input
value = col1.selectbox("Select Customer Value", ("Low", "Medium", "High", "Very High", "Top"))
region = col1.selectbox("Select Customer Region", ("North East", "North West", "North Central",
                                                    "South East", "South West", "South South"))
gender = col1.selectbox("Select Customer Gender", ("Male", "Female"))
d_type = col1.selectbox("Select Device Type", ("Smartphone", "Feature Phone"))


# If button is pressed
if st.button("Submit"):
    
    # Unpickle models
    binary = joblib.load("binary.pkl")
    sc = joblib.load("sc.pkl")
    rfc = joblib.load("rfc.pkl")

    
    # Store inputs into dataframe
    X = pd.DataFrame([[total, voice, data, sms, data_talk_rev, age, date, 
                        time, cus_cls, value, region, gender, d_type]], 
                     columns = ['spend_total','spend_voice','spend_data' ,'sms_cost','xtra_data_talk_rev',
                         'age', 'date','time',
                       'customer_class','customer_value','location_region','gender','device_type'])

    X['gender'] = X['gender'].map({'Male':'M', 'Female':'F'})                   
    X['day'] = [t.date().strftime('%A') for t in pd.to_datetime(X['date'])]
    X['period'] = ['day' if i.hour in range(6, 21) else 'night' for i in X['time']]


    X['age_group'] = ['1 - 12' if i in range(1,13)
                        else '13 - 16' if i in range(13,17)
                        else '17 - 40' if i in range(17,41)
                        else '41 - 64' if i in range(41,65)
                        else '65+'
                        for i in X.age]
    
    X = X[['customer_class',
            'gender',
            'device_type',
            'period',
            'day',
            'xtra_data_talk_rev',
            'spend_voice',
            'age_group',
            'sms_cost',
            'location_region',
            'spend_data',
            'customer_value',
            'spend_total']]


    # Get prediction
    X = binary.transform(X)
    X = sc.transform(X)
    prediction = rfc.predict(X)[0]
    predp = rfc.predict_proba(X)[:,1]
    q = ('Click' if prediction == 1 else 'sms')
    
    # Output prediction
    st.caption(f"This instance is a {q} with {round(predp[0] * 100, 2)}% chance")