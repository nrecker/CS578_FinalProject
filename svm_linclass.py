#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:57:57 2020

@author: anirban
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import svm

#Preprocessing function 
def preprocessData(data):
    data = data.select_dtypes(exclude=['object'])   #Excluding object data types
    data.fillna(data.median(), inplace = True)       #Filling missing data with median
    data.loc[(data.loss > 0),'loss'] = 1         #Replace all values >0 in loss by 1
    return data


data =pd.read_csv('/home/anirban/cs578/Final_project/loan-default-prediction/train_v2.csv')

#Create 10k data for just doing whatever. Not sure if this is a good ML approach 
#Correct me wherever you feel like . Here I am just trying to see how good accuraccy we get for just classification
#I set loss >0 to 1 and let 0 remain as they are
a=np.arange(105470)
np.random.shuffle(a)

reduced_train_data = data.iloc[a[0:10000]]
reduced_test_data = data.iloc[a[10000:20000]]

#Training with linear SVM
reduced_train_data = preprocessData(reduced_train_data)
X= reduced_train_data.drop(columns=['loss', 'id'])     #Since loss and id arent part of features
y= reduced_train_data['loss']
svm_linear = Pipeline([
        ("scaler", MinMaxScaler()),
        ("linear_svc", svm.LinearSVC(C=10, loss='squared_hinge', tol=1e-3, max_iter= 2000, dual=False))])
svm_linear.fit(X,y)


reduced_test_data = preprocessData(reduced_test_data)
X_test= reduced_test_data.drop(columns=['loss', 'id'])
y_test= reduced_test_data['loss']
y_pred= svm_linear.predict(X_test)

print (sum(abs(y_test-y_pred)))
print (len(np.where(y_test == 1)[0]))    #Place where loss is > 0  

