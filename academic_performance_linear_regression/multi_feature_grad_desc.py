import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

cur_dir = Path(__file__).parent
df = pd.read_csv(cur_dir /'Student_Performance.csv')

df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes':1, 'No':0})

X= df.iloc[:, :-1].to_numpy()
Y = df.iloc[:, -1].to_numpy()
print(df.head)

#normalizing X
mu = X.mean(axis=0)
sigma = X.std(axis=0)
X = (X - mu) / sigma


print(X.shape)
print(Y.shape)

def f_wb(w,b,x):
    
    return np.dot(x,w) + b

def cost(x, y, w_cur, b_cur):

    m = x.shape[0]
    err = np.zeros(m,)
    for i in range(m):
        err[i] = (f_wb(w_cur, b_cur, x[i])-y[i])**2
    return np.sum(err) / (2*m)


def dj_cost(x, y, w_cur, b_cur):

    m,n = x.shape
    err = np.zeros(m,)
    for i in range(m):
        err[i] = f_wb(w_cur, b_cur, x[i])-y[i]

    dw = np.zeros(n)
    for i in range(n):
        dw[i] = np.dot(err, x[:,i])
        dw[i] = dw[i] / m

    db = np.sum(err)/m

    return dw, db


def compute_grad_desc(x_train, y_train, w,b, alpha,iter):

    for i in range(iter):
        if i % 1000 == 0:
            print(f'Current cost:{cost(x_train, y_train, w, b)} \n w = {w}, b = {b}')

        dj_dw, dj_db =dj_cost(x_train, y_train, w, b) 
        dj_dw= alpha* dj_dw
        w = w - dj_dw
        b = b - alpha * dj_db
    return w, b


w_init = np.array([6, 1, 5, 5, 5])
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

y_pred = compute_pred(X, w_fin, b_fin)


#visualizing features
fig, axes=plt.subplots(1,6, figsize=(12,4))
n = X.shape[1]
for i in range(n):
    colors=['r','b','m','g', 'y']
    axes[i].scatter(X[:,i], Y, c= colors[i])


axes[5].scatter(Y, y_pred, s=8, alpha=0.4)
lims = [min(Y.min(), y_pred.min()), max(Y.max(), y_pred.max())]
axes[5].plot(lims, lims, 'r--', label='perfect prediction')
axes[5].set_xlabel('Actual y'); axes[5].set_ylabel('Predicted y'); plt.legend()
plt.show()
