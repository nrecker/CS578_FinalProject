#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:40:37 2020

@author: anirban
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import svm
import scipy
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score

#This code does feature selection using mutual information and 
#making new features from 2 features 
#Pre scaling not requd since mutual info not affected be scaling 
#Assumes y is 0 or 1 i.e classification case
def get_n_bestFeatures(X, y, n, combine =0):
    score = mutual_info_classif(X,y)
    colName = X.columns
    #Sort both score and colName the features according to score 
    score, colName = zip(*sorted(zip(score, colName)))
    #If combination of features not requd return the best n features
    if(combine == 0):
        return colName[-n:], score[-n:]
    else:
        #We will consider on the best n features and their combinations
        #So we have n^2 possibility and we select the best n 
        #Redundant features are an issue and best resolved by human eye
        newFeatureList = colName[-n:]
        
        finalScore = np.zeros(n)    #Holds scores for best n combined features
        finalFeature = ['None' for i in range(n)] #Holds combinations for best n combined features
        #Loop over all possible features
        for feature1 in newFeatureList:
            for feature2 in newFeatureList:
                
                #Comput all possible new features (add subtract and multiply. Division 
                #makes life difficult because of 1/0)
                new_add_feature = X[feature1] + X[feature2]
                new_sub_feature = X[feature1] - X[feature2]
                new_mul_feature = X[feature1] * X[feature2]
                
                #Find their information w.r.t y
                score_add = mutual_info_classif(new_add_feature.to_frame(),y)
                score_sub = mutual_info_classif(new_sub_feature.to_frame(),y)
                score_mul = mutual_info_classif(new_mul_feature.to_frame(),y)
                #If combination already considered continue
                if(score_add in  finalScore or score_sub in  finalScore or score_mul in  finalScore):
                    continue
                
                #Id combination not present add to finalScore and finalFeature
                for j in range(n):
                    if(score_add > finalScore[j]):
                        finalScore[j] = score_add
                        finalFeature[j] = str(feature1)+ ' + ' +str(feature2)
                        break
                for j in range(n):
                    if(score_sub > finalScore[j]):
                        finalScore[j] = score_sub
                        finalFeature[j] = str(feature1)+ ' - ' +str(feature2)
                        break
                for j in range(n):
                    if(score_mul > finalScore[j]):
                        finalScore[j] = score_mul
                        finalFeature[j] = str(feature1)+ ' * ' +str(feature2)
                        break
                    
        return finalFeature, finalScore
    

# =============================================================================
# #Tried to find mutual info between combination columns returned. Does not seem to work well
# #For eg between f743+f432 and f345+f767
# def calculateDependency(X,y,finalFeatureList):
#     n = len(finalFeatureList)
#     depMatrix = np.zeros((n,n))
#     for j in range(n):
#         for k in range(n):
#             f1_1 = finalFeatureList[j].split() [0]
#             operation1 =  finalFeatureList[j].split() [1]
#             f2_1 =  finalFeatureList[j].split() [2]
#             
#             f1_2 = finalFeatureList[k].split() [0]
#             operation2 =  finalFeatureList[k].split() [1]
#             f2_2 =  finalFeatureList[k].split() [2]
#             #print (f1_1, operation1, f1_2)
#             if(operation1 == '+'):
#                 combinedFeature1 = X[f1_1] + X[f2_1]
#             if(operation1 == '-'):
#                 combinedFeature1 = X[f1_1] - X[f2_1]
#             if(operation1 == '*'):
#                 combinedFeature1 = X[f1_1] * X[f2_1]
#                 
#             if(operation2 == '+'):
#                 combinedFeature2 = X[f1_2] + X[f2_2]
#             if(operation2 == '-'):
#                 combinedFeature2 = X[f1_2] - X[f2_2]
#             if(operation2 == '*'):
#                 combinedFeature2 = X[f1_2] * X[f2_2]
#             #print ('aa')    
#             score = mutual_info_score(combinedFeature1.to_frame(),combinedFeature1)
#             depMatrix[j,k] = score
#     return depMatrix
# =============================================================================
            
        
        

def preprocessData(data):
    data = data.select_dtypes(exclude=['object'])   #Excluding object data types
    data.fillna(data.median(), axis =0 ,inplace = True)       #Filling missing data with median
    data.loss.replace(np.arange(1,1000), 1, inplace=True)   
    return data


        
data =pd.read_csv('/home/anirban/cs578/Final_project/loan-default-prediction/train_v2.csv')
a=np.arange(105470)
reduced_train_data = data.iloc[a[20000:40000]]
reduced_train_data = preprocessData(reduced_train_data)
#X= reduced_train_data.drop(columns=['loss', 'id'])     #Since loss and id arent part of features
#Use soemthing like below to remove highly redundant features such as 'f527' and 'f528'
X= reduced_train_data.drop(columns=['loss', 'id','f275','f274','f135','f527','f528'])
y= reduced_train_data['loss']
fList, score = get_n_bestFeatures(X,y, 10)  #To get best 10 features
fList, score = get_n_bestFeatures(X,y, 0 ,0)  #To get best 10 combined features



