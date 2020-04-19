#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 09:18:10 2020

@author: anirban
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:58:14 2020

@author: anirban
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:32:48 2020

@author: anirban
"""

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
accuracyArr =[]
degree =2
pca_features = np.arange(5,60, 5) #Use only when using PCA
meanArr=[]
for features in pca_features:
        [reduced_train_data, reduced_test_data] = getdata(-1, k, data, 0)
        
        #Training with linear SVM
        reduced_train_data = preprocessData(reduced_train_data)
        X= reduced_train_data.drop(columns=['loss', 'id'])     #Since loss and id arent part of features
        y= reduced_train_data['loss']
        svm_linear = Pipeline([
                ("scaler", MinMaxScaler()),
                ("pca", PCA(n_components=features)),
                ('linear_svc', svm.SVC(kernel="poly", degree= degree, C=10, max_iter= 2000))])
                #("linear_svc", svm.LinearSVC(C=1, loss='squared_hinge', tol=1e-3, max_iter= 1000, dual=False))])
        svm_linear.fit(X,y)
        
        #Test
        reduced_test_data = preprocessData(reduced_test_data)
        X_test= reduced_test_data.drop(columns=['loss', 'id'])
        y_test= reduced_test_data['loss']
        y_pred= svm_linear.predict(X_test)
        
        accuracy = getScore(y_pred, y_test)
        accuracyArr.append(accuracy)
        
    
#Plot results
plt.plot(pca_features,accuracyArr)
plt.xlabel("No of Pca Feature")
plt.ylabel("Mean absolute error")
plt.title("Linear Kernel, C=1")
plt.show()