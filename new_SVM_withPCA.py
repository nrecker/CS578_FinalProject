#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:07:36 2020

@author: anirban
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import svm
import scipy
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import Ridge
#List of 63 best features I thought was useful. Obtained from feature select.py and 
#human analysis of removing redundant features 
fList = ['f768','f22','f21','f395','f25','f60','f339','f244','f397','f314','f270','f767','f406','f281','f26','f323',
 'f260','f772','f283','f675','f671','f405','f629','f27','f30','f18','f404','f29','f333','f764','f402','f332','f32',
 'f415','f736','f766','f282','f322','f400','f335','f337','f205','f135','f336','f136','f204','f274','f275','f527','f528',
 'f274 - f528','f528 - f274','f527 - f528','f528 * f275','f528 + f275','f528 + f528','f528 * f528','f527 * f274',
 'f527 + f528','f527 * f528','f336 - f337','f135 - f136','f204 + f768','f204 * f204','f336 + f629']

#This function will insert a new column by name of the combined features 
#in X and return X. For eg it would insert a column 'f527 + f528' in X if that is 
#present in f list
def getnewX(X, fList):
    #First add combined featues to X
    for feature in fList:
        if('+' in feature):
            sub_feature1 = feature.split()[0]
            sub_feature2 = feature.split()[2]
            X[feature] = X[sub_feature1] +  X[sub_feature2]
        if('-' in feature):
            sub_feature1 = feature.split()[0]
            sub_feature2 = feature.split()[2]
            X[feature] = X[sub_feature1] -  X[sub_feature2]
        if('*' in feature):
            sub_feature1 = feature.split()[0]
            sub_feature2 = feature.split()[2]
            X[feature] = X[sub_feature1] *  X[sub_feature2]
            
    new_X = X[fList]
    return new_X


            
#Preprocessing function  for classification
def preprocessData_classf(data):
    data = data.select_dtypes(exclude=['object'])   #Excluding object data types
    data.fillna(data.median(), axis =0 ,inplace = True)       #Filling missing data with median
    data.loss.replace(np.arange(1,1000), 1, inplace=True)
    return data    

#Preprocessing function for linear regression data
#It removoves all zero losses since a bunch of zeros will bias lin regression
def preprocessData_linreg(data):
    data = data.select_dtypes(exclude=['object'])   #Excluding object data types
    data.fillna(data.median(), axis =0 ,inplace = True)       #Filling missing data with median
    indexNames = data[ data['loss'] == 0 ].index
    data.drop(indexNames , inplace=True)
    return data    


#**************************************************************************************
#Modify below to include this code using k-fold.py output
#**********************************************************************************
data =pd.read_csv('/home/anirban/cs578/Final_project/loan-default-prediction/train_v2.csv')
a=np.arange(105470)
reduced_val_data = data.iloc[a[80000:100000]]
reduced_train_data = data.iloc[a[10000:60000]]
#Not using index.rest might cause some weird problems. Index for reduced_val_data
# for reduced_val_data = data.iloc[a[20000:40000]] starts from 20k
reduced_val_data=reduced_val_data.reset_index(drop=True)
reduced_train_data= reduced_train_data.reset_index(drop=True)

temp_reduced_val_data = reduced_val_data
temp_reduced_train_data = reduced_train_data
#**************************************************************************************
#END of data aquisition
#**************************************************************************************



#************************************************************************************
#This is the SVM Classification Part
#*************************************************************************************
reduced_train_data = preprocessData_classf(reduced_train_data)#ALERT!!Preprocess as per clsassf preprocess 
X= reduced_train_data.drop(columns=['loss', 'id'])     #Since loss and id arent part of features
#Get the new features
X_new = getnewX(X, fList)
y_classf= reduced_train_data['loss']
svm_linear = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=50)),
        #("linear_svc", svm.LinearSVC(C=10, penalty="l2", max_iter= 1000, dual=False))])
        ('linear_svc', svm.SVC(kernel="poly", C=1, degree=2, coef0=2, cache_size=1000))]) #
svm_linear.fit(X_new,y_classf)


y_actual = reduced_val_data['loss']
reduced_val_data = preprocessData_classf(reduced_val_data)
X_val= reduced_val_data.drop(columns=['loss', 'id'])
#Get the new features
X_val_new= getnewX(X_val, fList)
y_val_classf= reduced_val_data['loss']

y_pred= svm_linear.predict(X_val_new)

#No fo true/false positives/negatives etc etc 
#Feel free to change != 0 to >0 below or vice vesra
tp=0
fp=0
tn=0
fn =0
for j in range(len(y_pred)):
    if(y_pred[j] == 0 and y_val_classf[j]==0):
        tn += 1
    elif(y_pred[j] > 0 and y_val_classf[j] > 0):
        tp += 1
    elif(y_pred[j] == 0 and y_val_classf[j] >0):
        fn += 1
    elif(y_pred[j] > 0 and y_val_classf[j] ==0):
        fp += 1
    else:
        print ('HUH? ')
#**************************************************************************************
#END of SVM Part
#**************************************************************************************

#************************************************************************************
#This is the Regression prediction Part
#*************************************************************************************
poly_degree = 1  #Change here for polynomial regression. 
reduced_train_data = temp_reduced_train_data
reduced_val_data = temp_reduced_val_data
reduced_val_data=reduced_val_data.reset_index(drop=True)
reduced_train_data= reduced_train_data.reset_index(drop=True)
#***************************************************************************************
#End of data acquisition part
#**************************************************************************************

reduced_train_data = preprocessData_linreg(reduced_train_data) #ALERT!!Preprocess as per lin reg preprocess 
X= reduced_train_data.drop(columns=['loss', 'id'])     #Since loss and id arent part of features
X_new = getnewX(X, fList)
y_linReg= reduced_train_data['loss']
lin_reg_model = Pipeline([
         ("scaler", StandardScaler()),
         ("pca", PCA(n_components=10)),
         ("Poly Features", PolynomialFeatures(degree= poly_degree, include_bias= False )),
         ("linear_regression", Ridge(alpha= 10))])
    
lin_reg_model.fit(X_new, y_linReg)

zero_idx = np.where(y_pred==0)[0] 
X_val_linreg = X_val_new.drop(zero_idx)   #Drop all rows with loss = 0
y_pred_linreg = lin_reg_model.predict(X_val_linreg)

k=0
for j in range(len(y_pred)):
    if(y_pred[j] > 0):
        y_pred[j] = y_pred_linreg[k]   #When svm prediction is >0 replace it with lin reg prediction.
        k = k+1
print (np.mean(np.abs(y_actual-y_pred)))  #Print the mean abs error
#**************************************************************************************
#END of linear regression Part
#**************************************************************************************