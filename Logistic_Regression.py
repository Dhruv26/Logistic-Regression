import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

"""
x is the feature vector
y is the target vector
theta is the vector containing the weights of the features
Size:
x = [m x (n + 1)]
theta = [(n + 1) x 1]
y = [m * 1]
gradient = [(n + 1) x 1]
"""
"""
def sigmoid(x, theta):
  z = np.dot(np.transpose(theta), x)
  return 1 / (1 + np.exp(-z))
"""
# Choose the value of lambda
lamb = 10

def featureScaling(x):
  m = x.shape[0]
  n = x.shape[1]
  mean = x.mean(0)
  sigma = x.std(0)
  x_norm = (x - mean) / sigma
  return x_norm

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def costFunction(x, y, theta):
  h = sigmoid(np.dot(x, theta))
  m = x.shape[0]
  # Creating new variable for the regularized term
  newTheta = np.zeros((n, 1))
  for i in range(n):
    newTheta[i] = theta[i + 1]
  # To get a number to add we take the dot product of t' and t
  newTheta = np.dot(newTheta.transpose(), newTheta)
  j = (-1 / m) * (np.dot(np.transpose(y), np.log(h)) + np.dot((1 - y).
                  transpose(), np.log(1 - h))) + (lamb / (2 * m)) * newTheta
  return j

def gradient(x, y, theta):
  h = sigmoid(np.dot(x, theta))
  m = x.shape[0]
  n = x.shape[1]
  grad = (1 / m) * np.dot(np.transpose(x), (h - y))
  for i in range(1, n):
    grad[i] = grad[i] + (lamb / m) * theta[i]
  return grad

"""
x = np.array([np.random.randint(0, 10, size = 20)]).transpose()
y = np.array(np.random.randint(2, size = 20))
m = x.shape[0]
n = x.shape[1]
# Adding intercept term to x
x = np.column_stack((np.ones((m, 1)), x))
theta = np.zeros((n + 1, 1))
print(sigmoid(x))
print(y, theta.shape)
"""
# Inputing data from file
data = pd.read_csv("C:/pyt/ML/ex2data1.txt")

# Extracting x and y from the pandas dataframe
y = data['target']
x = data.loc[:, 'feature1': 'feature2']
x = featureScaling(x)

# Plotting the data
pos = data.loc[data['target'] == 1]
neg = data.loc[data['target'] == 0]
plt.scatter(pos['feature1'], pos['feature2'], c = 'b', marker = 'o')
plt.scatter(neg['feature1'], neg['feature2'], c = 'r', marker = 'x')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend(['Admitted', 'Not Admitted'], loc = 'upper right')
plt.show()

# Converting x to a numpy array Adding intercept term to x
y = np.array([y]).transpose()
x = x.as_matrix()
m = x.shape[0]
n = x.shape[1]
x = np.column_stack((np.ones((m, 1)), x))

# Initializing theta
initial_theta = np.zeros(((n + 1), 1))
"""
# Optimizing theta using minimize function in scipy module
result = op.minimize(fun = costFunction, x0 = initial_theta, args = (x, y),
                     method = 'Newton-CG', jac = gradient)
"""

# Implementing Gradient Descent
for i in range(1000):
  initial_theta = initial_theta - gradient(x, y, initial_theta)

