#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:34:01 2020

@author: anirban
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:35:53 2020

@author: anirban
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import svm
from kfold import *
from score import *
import matplotlib.pyplot as plt

#Preprocessing function 
def preprocessData(data):
    data = data.select_dtypes(exclude=['object'])   #Excluding object data types
    data.fillna(data.median(), inplace = True)       #Filling missing data with median
    data.loc[(data.loss > 0),'loss'] = 1         #Replace all values >0 in loss by 1
    return data


data =pd.read_csv('/home/anirban/cs578/Final_project/loan-default-prediction/train_v2.csv')
k=10
pca_features = 50 #Use only when using PCA
estimators = 50
rate =0.01
[reduced_train_data, reduced_test_data] = getdata(-1, k, data, -1)
    
#Training with linear SVM
reduced_train_data = preprocessData(reduced_train_data)
X= reduced_train_data.drop(columns=['loss', 'id'])     #Since loss and id arent part of features
y= reduced_train_data['loss']
        
adaboost = Pipeline([
            ("pca", PCA(n_components=pca_features)),
            ('adaboost',AdaBoostClassifier(n_estimators=estimators, learning_rate=rate))])
    
adaboost.fit(X,y)

#For bias and variance of test data
reduced_test_data = preprocessData(reduced_test_data)
X_test= reduced_test_data.drop(columns=['loss', 'id'])
y_test= reduced_test_data['loss']
y_pred= adaboost.predict(X_test)


print ('Test Data Bias Sq and Variance = '+str(getbias_var(y_test, y_pred)))
#For bias and variance of training data
y_pred= adaboost.predict(X)

biasSq =( np.mean(y) - np.mean(y_pred))**2
variance = np.std(y_pred) **2
print ('Training Data Bias Sq and Variance = '+str(biasSq) + ' '+str(variance))

