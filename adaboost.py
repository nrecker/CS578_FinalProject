#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostRegressor

from kfold import *
from score import *

#Preprocessing function 
def preprocessData(data):
    data = data.select_dtypes(exclude=['object'])   #Excluding object data types
    data.fillna(data.median(), inplace = True)       #Filling missing data with median
    data.loc[(data.loss > 0),'loss'] = 1         #Replace all values >0 in loss by 1
    return data


data =pd.read_csv('train_v2.csv')
estimators = 50
rate = 1
k =10
accuracyArr =[]
#pca_features = 40 #Use only when using PCA

for k_current in range(k):

    [reduced_train_data, reduced_val_data, reduced_test_data] = getdata(0, 10, data)

    #Training with adabost
    reduced_train_data = preprocessData(reduced_train_data)
    X= reduced_train_data.drop(columns=['loss', 'id'])     #Since loss and id arent part of features
    y= reduced_train_data['loss']
    #svm_linear = Pipeline([
    #        ("scaler", MinMaxScaler()),
    #        ("linear_svc", svm.LinearSVC(C=10, loss='squared_hinge', tol=1e-3, max_iter= 2000, dual=False))])
    adaboost = Pipeline([
            #("pca", PCA(n_components=pca_features)),
            ('adaboost',AdaBoostRegressor(n_estimators=estimators, learning_rate=rate))])
    adaboost.fit(X,y)


    reduced_val_data = preprocessData(reduced_val_data)
    X_test= reduced_val_data.drop(columns=['loss', 'id'])
    y_test= reduced_val_data['loss']
    y_pred= adaboost.predict(X_test)
    accuracy = getScore(y_pred, y_test)
    accuracyArr.append(accuracy)
    
mean, variance = getkFoldScore(accuracyArr)
print (mean, variance)

