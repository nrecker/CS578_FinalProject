import numpy as np
import csv

# Removes column labels
# Replaces NA with median for that feature
# Reads from train_v2.csv in the current directory
# Writes to a new csv, so it only needs to be run once ever
def doctor():
    n = 105471  #number of data
    d = 769     #number of features
    data = np.zeros((d,n));
    indices = np.zeros(d, dtype=int);
    with open('train_v2.csv',newline='') as infile:
        infile.readline() #skip column labels
        datareader = csv.reader(infile, delimiter=',', quotechar='|')
        j=0
        for row in datareader:
            j += 1
            if j%100 == 0:
                print(j)
            for i in range(d):
                if row[i] != 'NA':
                    data[i][indices[i]] = float(row[i])
                    indices[i] += 1
                    
    medians = np.zeros(d)
    for i in range(d):
        actualData = data[i][:indices[i]]
        actualData.sort()
        medians[i] = actualData[ len(actualData)//2 ]
        
    with open('train_v2.csv',newline='') as infile:
        with open('train_v2_doctored.csv', 'w', newline='') as outfile:
            infile.readline() #skip column labels
            datareader = csv.reader(infile, delimiter=',', quotechar='|')
            datawriter = csv.writer(outfile, delimiter=',', quotechar='|')
            i = 0
            for row in datareader:
                i += 1
                if i%100 == 0:
                    print(i)
                for i in range(d):
                    if row[i] == 'NA':
                        row[i] = str(medians[i])
                datawriter.writerow(row)
        
# Returns (Xtrain,ytrain, Xtest,ytest)
# Xtrain's dimensions are (n*9/10, d) where n,d are number of samples and number of features
# ytrain's dimensions are (n*9/10, 1)
# Xtest's dimensions are (n/10, d)
# ytest's dimensions are (n/10, 1)
#
# seed is for random separation of data; I think we should always use the same seed 
def getdata(seed=0):
    n = 105471
    d = 769

    # Read data
    X = np.empty((n,d))
    y = np.empty((n,1))
    with open('train_v2_doctored.csv',newline='') as file:
        datareader = csv.reader(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
        i = 0
        for row in datareader:
            X[i] = row[1:-1]
            y[i] = row[-1]
            i += 1
    
    # Divide into training/validation and testing
    np.random.seed(seed)
    testSubset = np.random.choice(n,n//10, replace=False)
    trainSubset = np.setdiff1d( np.array(range(n)), testSubset )
    return X[trainSubset], y[trainSubset],  X[testSubset], y[testSubset]

#doctor()

Xtrain, ytrain, Xtest, ytest = getdata()
print(Xtrain[1][2])