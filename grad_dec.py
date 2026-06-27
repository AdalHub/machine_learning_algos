import numpy as np
import matplotlib.pyplot as plt
import math
#sqft
x_train = np.array([3812, 4156, 3442,2008, 4358, 1820])


#cost in 1000s
y_train = np.array([830, 659, 875, 400, 859, 374])

def f_wb(w,b,x_train):
    
    m =x_train.shape[0]
    res= np.zeros(m)
    
    for i in range(m):
        y= w*x_train[i] + b
        res[i]= y

    return res

def cost(yhat, y_train):

    m = yhat.shape[0]
    curCost = 0
    for i in range(m):
        curCost += (yhat[i]-y_train[i])**2
    return (curCost/(m*2))

def derJw(w,b,x_train,y_train)-> int:
    m =x_train.shape[0]
    res= np.zeros(m)
    curCost = 0
    for i in range(m):
        curCost += ((w*x_train[i] + b) - y_train[i])*x_train[i]

    return curCost/m

def derJb(w,b,x_train,y_train)-> int:
    m =x_train.shape[0]
    res= np.zeros(m)
    curCost = 0
    for i in range(m):
        curCost += (w*x_train[i] + b) - y_train[i]

    return curCost/m


#since b and w are vastly different in scale we create two alpha values
alpha_w =1e-7
alpha_b= 1e-1
costIter= 10000
curW, curB = .02, 100

for i in range(costIter):
    yPred = f_wb(curW,curB,x_train)
    print(f"CURRENT W = {curW}, CURRENT B = {curB}")
    print(f"current cost { cost(yPred, y_train)}")
    frozenW, frozenB= curW, curB
    print(curW)
    print(f"gradW = {derJw(frozenW, frozenB, x_train, y_train)}, gradB = {derJb(frozenW, frozenB, x_train, y_train)}")
    curW= curW - alpha_w * derJw(frozenW,frozenB,x_train, y_train )
    curB = curB - alpha_b * derJb(frozenW,frozenB,x_train,y_train )

yPred = f_wb(curW,curB,x_train)
print(f"FINAL W = {curW}, FINAL B = {curB}")
print(f"FINAL cost{ cost(yPred,y_train= y_train)}")
plt.scatter(x_train, y_train, marker= "X", c= "b")
order = np.argsort(x_train)
plt.plot(x_train[order], yPred[order], c="r")
plt.show()
