from sklearn import svm
from kfold import *

maxk = 10
Xtrain,ytrain, Xval,yval, Xtest,ytest = getdata(0,10)

print('check 1')

print(sum(yval))

clf = svm.LinearSVC(max_iter=10000)
print('check 2')
clf.fit(Xtrain,ytrain)
print('check 3')
pred = clf.predict(Xval)
print(pred[:10])
print(sum(abs( pred-yval )))
