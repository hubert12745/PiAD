import numpy as np
import scipy.sparse as sp
import pandas as pd

data = pd.read_csv("zoo.csv")
#print(data)

def freq(X, prob = True):
    Pi = {}
    N = X.shape[0]
    for x in X:
        if x in Pi:
            Pi[x]+=1
        else:
            Pi[x] = 1
    if prob == True:
        for p in Pi:
            Pi[p]/=N
    return Pi

print(freq(data['type']))

def freq2(X, Y, prob=True):
    xi = {}
    yi = {}
    ni = {}
    N = X.shape[0]
    for x,y in zip(X,Y):
        if x in xi:
            xi[x]+=1
        else:
            xi[x] = 1
        if y in yi:
            yi[y]+=1
        else:
            yi[y] = 1

    for xy in zip(X,Y):
        if xy in ni:
            ni[xy]+=1
        else:
            ni[xy] = 1
    if prob:
        for x in xi:
            xi[x] /= N
        for y in yi:
            yi[y] /= N
        for n in ni:
            ni[n] /= N

    return xi,yi,ni



print(freq2(data['hair'],data['feathers']))

def entropy(x):
    H = np.sum(x*np.log2(x))
    return H
def infogain(x,y):
    I = entropy(freq(x)) + entropy(freq(y)) - entropy(freq2(x,y))
    return I
print(infogain(data['hair'],data['feathers']))