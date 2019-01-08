import numpy as np
from sslMarkovRandomWalks import sslMarkovRandomWalks
# import dataset into X as input and y as label matrix
from sklearn.datasets import load_iris


np.random.seed(seed=1234)

iris = load_iris()
X = iris.data  
y = iris.target

# suppose 0-20th data are labelled as 1 and 50-70th data are labelled as -1
I=np.zeros(X.shape[0],dtype='bool')
I[0:20]=True
I[50:70]=True
xl=X[I,:]
yl=-2*(y[I]-0.5)

# suppose 20-50th lablled as 1 and 70-100th data are labelled as -1 but these are unknown
J=np.zeros(X.shape[0],dtype='bool')
J[20:50]=True
J[70:100]=True
xu=X[J,:]
yu=-2*(y[J]-0.5)

# run random walk by known and unknown data and predict label of unknown data
yhatu,prob=sslMarkovRandomWalks(xl,yl,xu)
# check general accuracy of prediction
print('Prediction accuracy:', (yhatu==yu).mean())

print()
print('Each test elemen and its prediction probabilities')
for i in range(0,len(yhatu)):
    print('Actual class :',yu[i],' Probability of +1 class:',prob[i])
