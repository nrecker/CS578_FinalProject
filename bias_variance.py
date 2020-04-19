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
pca_features = 25 #Use only when using PCA

[reduced_train_data, reduced_test_data] = getdata(-1, k, data, -1)
    
#Training with linear SVM
reduced_train_data = preprocessData(reduced_train_data)
X= reduced_train_data.drop(columns=['loss', 'id'])     #Since loss and id arent part of features
y= reduced_train_data['loss']
svm_linear = Pipeline([
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=pca_features)),
        #("linear_svc", svm.LinearSVC(C=1, loss='squared_hinge', tol=1e-3, max_iter= 1000, dual=False))])
        ('linear_svc', svm.SVC(kernel="rbf", C=1,  max_iter= 1000, random_state=1))])
svm_linear.fit(X,y)

#For bias and variance of test data
reduced_test_data = preprocessData(reduced_test_data)
X_test= reduced_test_data.drop(columns=['loss', 'id'])
y_test= reduced_test_data['loss']
y_pred= svm_linear.predict(X_test)


print (getbias_var(y_test, y_pred))
#For bias and variance of training data
# =============================================================================
# y_pred= svm_linear.predict(X)
# 
# biasSq =( np.mean(y) - np.mean(y_pred))**2
# variance = np.std(y_pred) **2
# print (biasSq,variance)
# =============================================================================

