import csv
import numpy as np
from sslMarkovRandomWalks import sslMarkovRandomWalks


np.random.seed(seed=1234)

# read data , inputs are in X, labels are in Y
f=open('../data/voting.data')
X=[]
Y=[]
reader = csv.reader(f, delimiter=',')
for row in reader:
    tmp=[]
    for i in range(1,len(row)-1):        
        if row[i]=='+':
            tmp.append(1)
        elif row[i]=='-':
            tmp.append(-1)
        else:
            tmp.append(0)
    X.append(tmp)
    if row[-1]=='0':
        Y.append(-1)
    else:
        Y.append(1)

X=np.array(X)
Y=np.array(Y)


# to divide the dataset into two part as random one is labelled another part is unlabelled

# creaet a index from 0 to number of data and randomly reorder it
n=X.shape[0]
no2=int(n/5)
indices = np.arange(n)
np.random.shuffle(indices)

# divide it into the midlle train and test
Xtrain=X[indices[0:no2],:]
Ytrain=Y[indices[0:no2]]

Xtest=X[indices[no2:],:]
Ytest=Y[indices[no2:]]


# run method
yhat,prob=sslMarkovRandomWalks(Xtrain,Ytrain,Xtest)

# calculate accuracy
acc=(yhat==Ytest).mean()

print('Accuracy :'+str(acc))

