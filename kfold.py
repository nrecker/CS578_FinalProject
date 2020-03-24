import numpy as np
import csv

# Removes column labels
# Replaces NA with median for that feature
# Reads from train_v2.csv in the current directory
# Writes to a new csv, so it only needs to be run once ever
def doctor(maxk):
    n = 105471  #number of data
    d = 769     #number of features
    
    testStart = n - (n // 10)   #index of first row of test data; equivalently, number of training/validation data
    
    # Collect medians
    data = np.zeros((maxk, d, n//maxk)); #bigger than needed
    indices = np.zeros((maxk,d), dtype=int);
    with open('train_v2.csv',newline='') as infile:
        infile.readline() #skip column labels
        datareader = csv.reader(infile, delimiter=',', quotechar='|')
        j=0
        k=0
        for row in datareader:
            if j >= testStart:
                break
            if j%1000 == 0: #to mark progress
                print(j)
            for i in range(d):
                if row[i] != 'NA':
                    data[k, i, indices[k,i]] = float(row[i])
                    indices[k,i] += 1
            j += 1
            if j%(testStart//maxk) == 0 and k+1 < maxk:
                k += 1
                    
    medians = np.zeros((maxk,d))
    for k in range(maxk):
        for i in range(d):
            relevant = np.zeros(0)
            for kiter in range(maxk):
                if kiter == k:
                    continue
                relevant = np.concatenate((relevant, data[k, i, :indices[k,i]]))
            medians[k,i] = np.median(relevant)
    testMedian = np.zeros(d)
    for i in range(d):
        relevant = np.zeros(0)
        for kiter in range(maxk):
            relevant = np.concatenate((relevant, data[k, i, :indices[k,i]]))
        testMedian[i] = np.median(relevant) #we can use all training/validation data when testing
    
    # Write the doctored data
    with open('train_v2.csv',newline='') as infile:
        infile.readline() #skip column labels
        datareader = csv.reader(infile, delimiter=',', quotechar='|')
        outfiles = []
        writers = []
        for k in range(maxk):
            outfiles.append( open('train_v2_doctored_'+str(k)+'.csv', 'w', newline='') )
            writers.append( csv.writer(outfiles[-1], delimiter=',', quotechar='|') )
        
        j = 0
        for row in datareader:
            j += 1
            if j%1000 == 0: #to mark progress
                print(j)
            for k in range(maxk):
                rowcopy = row.copy()
                for i in range(d):
                    if rowcopy[i] == 'NA':
                        if j >= testStart:
                            rowcopy[i] = str(testMedian[i])
                        else:
                            rowcopy[i] = str(medians[k,i])
                writers[k].writerow(rowcopy)
        
        for file in outfiles:
            file.close()
        
# Returns (Xtrain,ytrain, Xtest,ytest for a particular value of k
# Requires that doctor() has previously been called to generate the necessary files
# Xtrain's dimensions are (n*9/10, d) where n,d are number of samples and number of features
# ytrain's dimensions are (n*9/10, 1)
# Xtest's dimensions are (n/10, d)
# ytest's dimensions are (n/10, 1)
def getdata(k):
    n = 105471
    d = 769

    # Read data
    X = np.empty((n,d))
    y = np.empty((n,1))
    with open('train_v2_doctored_'+str(k)+'.csv',newline='') as file:
        datareader = csv.reader(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
        i = 0
        for row in datareader:
            X[i] = row[1:-1]    #the first column is just an id number
            y[i] = row[-1]
            i += 1
    
    testStart = n - (n // 10)
    return X[:testStart], y[:testStart],  X[testStart:], y[testStart:]

# doctor(10)