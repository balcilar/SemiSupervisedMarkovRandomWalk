import numpy as np


def sslMarkovRandomWalks(xl,yl,xu,k=10,gamma=1,t=10,improvement=10e-5):
    # Written by Muhammet Balcilar, muhammetbalcilar@gmail.com,   France

    # xl is nxd size matrix shows inputs of known data. n is number of data d is dimension
    # yl is nx1 size vector shows label it is either 1 or -1
    # xu is mxd size matrix shows input of unknown data. m is number of data d is dimension
    # k shows number of k-nn which keep the most k number of closest neightbor of each data point
    # gamma for w function
    # t for degree of power
    # improvement is stopping criteria if there is less than improvement in labels difference


    # merge all inputs in one matrix
    x=np.vstack((xl,xu))
    knownlabel=len(yl)
    
    # calculate euclide distances for every pair of data point 
    d=np.zeros((x.shape[0],x.shape[0]),dtype='float')
    for i in range(0,x.shape[0]-1):
        for j in range(i+1,x.shape[0]):
            d[i,j]=np.sqrt(((x[i,:]-x[j,:])**2).sum())
            d[j,i]=d[i,j]

    # find most closest k data point for each data point
    index=[]
    for i in range(0,x.shape[0]):
        index.append(np.argsort(d[i,:])[0:k])

    # create w matrix
    w=np.zeros((x.shape[0],x.shape[0]),dtype='float')

    # fill w matrix according to paper
    for i in range(0,x.shape[0]):
        w[index[i],i]=np.exp(-d[index[i],i]/gamma**2)
    for i in range(0,x.shape[0]):
        w[i,i]=1

    # initial A matirx
    At=w/w.sum(axis=1)
    prob=At 

    # take t power of A matrix
    for i in range(1,t):
        At=At.dot(prob)

    #  set initial probabilities. +1 labelled data's P is 1, -1 labelled data's P is 0
    # unknonw data's P is 0.5 (means no information either -1 or 1)
    P=np.zeros((x.shape[0],1))
    Pold=P.copy()
    posindex=np.where(yl==1)[0]
    P[posindex]=1
    negindex=np.where(yl==-1)[0]
    uindex=[i for i in range(len(yl),x.shape[0])]
    P[uindex]=0.5

    cSums = At.sum(axis=0)
    #Expectation Maximization step
    while ((P-Pold)**2).sum() > improvement: # if tehre is significant improvement keep update of P
        # update mechanism of P
        Pold=P.copy()
        Ppos=At*P
        P[:,0]=Ppos.sum(axis=0)/cSums
        # even known data P is differen force them as known value
        # in that way we just change the unknonw data's P value 
        P[posindex,0]=1
        P[negindex,0]=0

    # make final decision 
    yu=P[:,0].copy()
    for i in range(0,P.shape[0]):# if P value is greater than 0.5 make it 1 unles -1
        if yu[i]>=0.5:
            yu[i]=1
        else:
            yu[i]=-1
    # return assigned label and probabilities
    return yu[uindex],P[uindex,0]

