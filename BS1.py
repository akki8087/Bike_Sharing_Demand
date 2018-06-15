# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:22:13 2018

@author: NP
"""



import numpy as np
import pandas as pd
import math

data = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')

X = data.iloc[:,1:-3]
y = data.iloc[:,-1]

X_d = pd.get_dummies(X, columns = ['season', 'holiday', 'workingday', 'weather'],drop_first = True)

'''
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_d, y, test_size = 0.2, random_state = 0)
'''
X_train = X_d
y_train = y
X_test = test.iloc[:,1:]
X_test = pd.get_dummies(X_test,columns = ['season', 'holiday', 'workingday', 'weather'],drop_first = True)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''



import xgboost as xgb
from xgboost.sklearn import XGBRegressor
regressor = XGBRegressor()



'''
# Fitting Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Random forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state = 0)
'''
'''
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
'''

# Predicting the Test set results
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


result = pd.DataFrame()
result['datetime'] = test['datetime']
result['count'] = y_pred
result.to_csv('BS2.csv',index = False)


#rms = math.sqrt(sum((y_pred - y_test)**2)/len(X_test))

