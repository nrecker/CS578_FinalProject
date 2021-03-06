import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import svm
from kfold import *
from score import *

#Preprocessing function 
def preprocessData(data):
    data = data.select_dtypes(exclude=['object'])   #Excluding object data types
    data.fillna(data.median(), inplace = True)       #Filling missing data with median
    data.loc[(data.loss > 0),'loss'] = 1         #Replace all values >0 in loss by 1
    return data


data =pd.read_csv('train_v2.csv')
k=10
accuracyArr =[]
degree = 2
#pca_features = 40 #Use only when using PCA

for k_current in range(k):
    
    [reduced_train_data, reduced_val_data, reduced_test_data] = getdata(k_current, k, data)
    
    #Training with linear SVM
    reduced_train_data = preprocessData(reduced_train_data)
    X= reduced_train_data.drop(columns=['loss', 'id'])     #Since loss and id arent part of features
    y= reduced_train_data['loss']
    svm_linear = Pipeline([
            ("scaler", MinMaxScaler()),
            #("pca", PCA(n_components=pca_features)),
            #('linear_svc', svm.SVC(kernel="rbf", C=1, max_iter= 1000))])
            ('linear_svc', svm.SVC(kernel="poly", C=1, degree=degree,max_iter= 1000))])
    svm_linear.fit(X,y)
    
    
    reduced_val_data = preprocessData(reduced_val_data)
    X_test= reduced_val_data.drop(columns=['loss', 'id'])
    y_test= reduced_val_data['loss']
    y_pred= svm_linear.predict(X_test)
    
    accuracy = getScore(y_pred, y_test)
    accuracyArr.append(accuracy)
    
mean, variance = getkFoldScore(accuracyArr)
print (mean, variance)
#874.4 2117.2400000000007    

