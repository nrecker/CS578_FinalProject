import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler

# Partitions full dataset into training, validation, and test data
# 0 <= k < maxk
def getdata(k, maxk, data):
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

#testing code
[a,b,c] = getdata(2,10,pd.read_csv('train_v2.csv'))