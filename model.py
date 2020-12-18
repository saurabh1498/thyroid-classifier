# -*- coding: utf-8 -*-

import pandas as pd
import pickle
df = pd.read_csv('full_dataset.csv')
df1=df[['sex','age','FTI','T4U','T3','TT4','TSH','classes']]

print("Enter your own data to test the model:")
age_n = float(input("age:"))
sex_n = float(input("sex:"))
FTI_n = float(input("FTI:"))
T4U_n = float(input("T4U:"))
T3_n = float(input("T3:"))
TT4_n = float(input("TT4:"))
TSH_n = float(input("TSH:"))

userInput =[age_n,sex_n,FTI_n,T4U_n,T3_n,TT4_n,TSH_n]
from sklearn.model_selection import train_test_split
feature_columns = ['sex','age','FTI','T4U','T3','TT4','TSH']
predicted_class = ['classes']

X = df1[feature_columns].values
y = df1[predicted_class].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)
random_forest_model.fit(X_train, y_train.ravel())
result = random_forest_model.predict([userInput])[0]

if(result==1):
    print("thyroid positive")
else:
    print("thyroid negative")
    
saved_model = pickle.dumps(random_forest_model)
random_forest_model_from_pickle = pickle.loads(saved_model) 
random_forest_model_from_pickle.predict(userInput) 
pickle.open(saved_model)