import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from pathlib import Path

cur_dir= Path(__file__).parent
df = pd.read_csv(cur_dir/'Salary_dataset.csv')
df.head()
df.shape         # (rows, cols) -> your (m, n+1)
df.columns       # column names
df.dtypes        # type per column
df.describe()    # count/mean/std/min/max per numeric column
df.info()  

X = df['YearsExperience'].to_numpy()
Y = df['Salary'].to_numpy()

def f_wb(w,b,x):
    return w*x+b

def cost(x, y, w_cur, b_cur):

    m = x.shape[0]
    err = np.zeros(m,)
    for i in range(m):
        err[i] = (f_wb(w_cur, b_cur, x[i])-y[i])**2
    return np.sum(err) / (2*m)


def dj_cost(x, y, w_cur, b_cur):

    m = x.shape[0]
    err = np.zeros(m,)
    for i in range(m):
        err[i] = f_wb(w_cur, b_cur, x[i])-y[i]

    dw = np.dot(x, err)
    dw = np.sum(dw)/m

    db = np.sum(err)/m

    return dw, db


def compute_grad_desc(x_train, y_train, w,b, alpha,iter):

    for i in range(iter):
        if i % 1000 == 0:
            print(f'Current cost:{cost(x_train, y_train, w, b)} \n w = {w}, b = {b}')

        dj_dw, dj_db =dj_cost(x_train, y_train, w, b) 
        w = w - alpha* dj_dw
        b = b - alpha * dj_db
    return w, b

w_init = 40000
b_init = 0
learning_rate = 1e-2
num_iter = 1000

w_fin, b_fin = compute_grad_desc(X, Y, w_init, b_init, learning_rate, num_iter)
print(f'Final cost:{cost(X, Y, w_fin, b_fin)} \n w = {w_fin}, b = {b_fin}')



def compute_pred(x_train, w, b):
    m =x_train.shape[0]
    y_pred = np.zeros(m)

    for i in range(m):
        y_pred[i]= f_wb(w,b,x_train[i])
    return y_pred

plt.scatter(X, Y, marker= "x", c= "r")
order = np.argsort(X)
plt.plot(X[order], compute_pred(X, w_fin, b_fin)[order])
plt.show()
