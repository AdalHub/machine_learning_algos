import numpy as np
import matplotlib.pyplot as plt


#cost of a zillow house
#sqft of houses
x_train = np.array([3812, 4156, 3442,2008, 4358, 1820])
#cost in 1000s
y_train = np.array([830, 659, 875, 400, 859, 374])

plt.scatter(x_train, y_train, marker= "x", c= "r")

plt.title("zillow houses in austin")
plt.xlabel("sqft of houses")
plt.ylabel("cost in 1000s")

#lets say our initial expectation is that houses roughly follow .16(w) + 75(bias)

def computeCost(w,b,x,y):
    m = x_train.shape[0]
    costs= []
    for i in range(m):
        #calculate y hat

        f_wb= w*x[i]+b
        tot_cost = (f_wb - y[i])**2
        costs.append(tot_cost)
    return (sum(costs)/ (m*2))


print(computeCost(.18, 75, x_train, y_train))





plt.show()
