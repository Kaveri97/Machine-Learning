import matplotlib.pyplot as plt
import numpy as np
import csv as csv
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_pred = regr.predict(diabetes_X_test)

print('coefficients: \n', regr.coef_)

plt.scatter(diabetes_X_test,diabetes_Y_test, color='black')
plt.plot(diabetes_X_test, diabetes_Y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

from numpy import * 
import matplotlib.pyplot as plt

points = genfromtxt('data.csv', delimiter=',')
x = array(points[:,0])
y = array(points[:,1])
plt.scatter(x,y)
plt.xlabel('Hours of study')
plt.ylabel('Test scores')
plt.title('Dataset')
plt.show()

learning_rate = 0.0001
initial_b = 0
initial_m = 0
num_iterations = 10

def compute_cost(b, m, points):
    total_cost = 0
    N = float(len(points))
   
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        total_cost += (y - (m * x + b)) ** 2
       
    return total_cost/N

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

b, m = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

print('Optimized b:', b)
print('Optimized m:', m)

print('Minmized cost:', compute_cost(b, m, points))

plt.scatter(x, y)
pred = m * x + b
plt.plot(x,pred,c='r')
plt.xlabel('Hours of study')
plt.xlabel('Test scores')
plt.title('Line of best fit')
plt.show()

