import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler

# Partitions full dataset into training, validation, and test data
# 0 <= k < maxk
def getdata(k, maxk, data, kFold_flag = 1):
    if(kFold_flag == 1):
        n = len(data)
        a=np.arange(n)
        nTrainVal = n - n//10
        nVal = nTrainVal//maxk  # k=maxk-1 is actually a little larger because of rounding
    
        if k == maxk-1: # handled separately because last k gets the roundoff
            train = data.iloc[a[:k*nVal]]
            val = data.iloc[a[k*nVal:nTrainVal]]
        else:
            train = data.iloc[np.concatenate((a[:k*nVal], a[(k+1)*nVal:nTrainVal]))]
            val = data.iloc[a[k*nVal:(k+1)*nVal]]
        
        test = data.iloc[a[nTrainVal:]]
            
        return train, val, test
    else:
        #When we want all the  training and test data and NO validation 
        n = len(data)
        a=np.arange(n)
        nTrainVal = n - n//10
        nVal = nTrainVal//maxk  # k=maxk-1 is actually a little larger because of rounding
        train = data.iloc[a[0:nTrainVal]]
        test = data.iloc[a[nTrainVal:]]
        
        return train,test

#testing code
[a,b,c] = getdata(2,10,pd.read_csv('train_v2.csv'))
