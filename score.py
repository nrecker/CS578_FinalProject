#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:14:57 2020

@author: anirban
"""

import numpy as np
import pandas as pd

#Returns the mae score
def getScore(y, y_pred):
    #Convert to numpy array for faster operation
    y = np.array(y)
    y_pred = np.array(y_pred)
    if(len(y) != len(y_pred)):
        print ('Length mismatch error!')
        return -1
    #This competition is evaluated on the mean absolute error (MAE)
    score = np.mean(np.abs(y-y_pred))
    return score

#Takes an array of mae scores for all ks for a given algorithm 
# and returns the mean and vairance
def getkFoldScore(scoreArr):
    scoreArr =np.array(scoreArr)
    mean = np.mean(scoreArr)
    
    variance = np.std(scoreArr) ** 2
    
    return mean, variance
    
def getbias_var(y_actual, y_pred):
    biasSq = (np.mean(y_actual) - np.mean(y_pred))**2    
    variance = np.std(y_pred) **2
    return biasSq,variance    
