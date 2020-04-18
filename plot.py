import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import svm
import numpy as np
from kfold import *
from score import *

def roc(pipeline, X_test, y_test):
    y_pred = pipeline.decision_function(X_test)
    y_pred = np.array(y_pred)
    
    threshes = y_pred.copy()
    threshes.sort()
    
    # Calculate True positive and False positive
    tp = np.zeros(threshes.size+1)
    fp = np.zeros(threshes.size+1)
    for i in range(threshes.size):
        tp[i] = sum(np.logical_and( y_test == 1, y_pred < threshes[i] ))
        fp[i] = sum(np.logical_and( y_test == 0, y_pred < threshes[i] ))
    # thresh = infinity
    tp[-1] = sum(y_test == 1)
    fp[-1] = sum(y_test == 0)
    
    tp = tp/sum(y_test == 1)
    fp = fp/sum(y_test == 0)
    
    # Plot ROC curve
    plt.plot(fp,tp)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.show()
    
    # Calculate AUROC
    area = 0
    for i in range(tp.size-1):
        area += (tp[i]+tp[i+1])/2 * (fp[i+1]-fp[i])   # Trapezoidal approximation
    return area

#testing code from here down
def preprocessData(data):
    data = data.select_dtypes(exclude=['object'])   #Excluding object data types
    data.fillna(data.median(), inplace = True)       #Filling missing data with median
    data.loc[(data.loss > 0),'loss'] = 1         #Replace all values >0 in loss by 1
    return data    
data =pd.read_csv('train_v2.csv')
[reduced_train_data, reduced_val_data, reduced_test_data] = getdata(1, 10, data)
reduced_train_data = preprocessData(reduced_train_data)
X= reduced_train_data.drop(columns=['loss', 'id'])     #Since loss and id arent part of features
y= reduced_train_data['loss']
svm_linear = Pipeline([
        ("scaler", MinMaxScaler()),
        #("pca", PCA(n_components=pca_features)),
        ("linear_svc", svm.LinearSVC(C=1, loss='squared_hinge', tol=1e-3, max_iter= 10, dual=False))])
svm_linear.fit(X,y)


reduced_val_data = preprocessData(reduced_val_data)
X_test= reduced_val_data.drop(columns=['loss', 'id'])
y_test= reduced_val_data['loss']
print( roc(svm_linear, X_test, y_test) )