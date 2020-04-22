import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
from sklearn import preprocessing
from sklearn import svm
import numpy as np
from kfold import *
from score import *

def roc(pipeline, X_test, y_test, color, message):
    y_pred = pipeline.decision_function(X_test)
    y_pred = np.array(y_pred)
    
    threshes = y_pred.copy()
    threshes.sort()
    
    # Calculate True positive and False positive
    tp = np.zeros(threshes.size+1)
    fp = np.zeros(threshes.size+1)
    for i in range(threshes.size):
        tp[i] = sum(np.logical_and( y_test == 1, y_pred >= threshes[i] ))
        fp[i] = sum(np.logical_and( y_test == 0, y_pred >= threshes[i] ))
    # thresh = infinity
    tp[-1] = 0
    fp[-1] = 0
    
    tp = tp/sum(y_test == 1)
    fp = fp/sum(y_test == 0)
    
    # Plot ROC curve
    plt.plot(fp,tp, color, label = message)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc='upper left')
    plt.show()
    # Calculate AUROC
    area = 0
    for i in range(tp.size-1):
        area += (tp[i]+tp[i+1])/2 * (fp[i]-fp[i+1])   # Trapezoidal approximation
    return area

#testing code from here down
def preprocessData(data):
    data = data.select_dtypes(exclude=['object'])   #Excluding object data types
    data.fillna(data.median(), inplace = True)       #Filling missing data with median
    data.loc[(data.loss > 0),'loss'] = 1         #Replace all values >0 in loss by 1
    return data 


learning_rates=[0.1,0.01, 0.001]
names =['Rate=0.1', 'Rate=0.01','Rate=0.001']
estimators = 30
size=3
areaArr=[]
color = ['b-', 'g-', 'r-', 'y-']
for j in range(size):  
    data =pd.read_csv('/home/anirban/cs578/Final_project/loan-default-prediction/train_v2.csv')
    [reduced_train_data, reduced_test_data] = getdata(1, 10, data, -1)
    reduced_train_data = preprocessData(reduced_train_data)
    X= reduced_train_data.drop(columns=['loss', 'id'])     #Since loss and id arent part of features
    y= reduced_train_data['loss']

    adaboost = Pipeline([
            ("pca", PCA(n_components=50)),
            ('adaboost',AdaBoostClassifier(n_estimators=estimators, learning_rate=learning_rates[j]))])
    adaboost.fit(X,y)

    
    
    reduced_test_data = preprocessData(reduced_test_data)
    X_test= reduced_test_data.drop(columns=['loss', 'id'])
    y_test= reduced_test_data['loss']
    areaArr.append(roc(adaboost, X_test, y_test, color[j], names[j]) )
    #area,tp,fp,threshes = roc(adaboost, X_test, y_test, color[j], names[j])
print (areaArr)
