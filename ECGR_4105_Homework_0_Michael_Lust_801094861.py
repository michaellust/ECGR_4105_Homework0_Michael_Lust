#!/usr/bin/env python
# coding: utf-8

# In[2182]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2183]:


df = pd.read_csv('D3.csv')
df.head()                      # To get first n rows from the dataset default value of n is 5
M = len(df)
M 


# In[2184]:


#Starting of with testing the explanatory value X1
X = df.values[:, 0] #Getting the values of each variable in the first column
Y = df.values[:, 3] #Setting the last column as my result for y.
M = len(Y)# Number of training examples
print('X = ', X[:99])   # All variables shown given there are M = 99 samples.
print('Y = ', Y[:99])
print('M = ', M)


# In[2185]:


plt.scatter(X,Y,color='blue',marker='o')
plt.grid()
plt.rcParams["figure.figsize"]=(10,6)
plt.title('Scatter plot of training data for Homework_0')


# In[2186]:


# Creating a matrix with single column of ones
X_0 = np.ones((M,1))
X_0[:5]


# In[2187]:


# Using reshape function convert X 1D array to 2D array of dimension 2X1
X_1 = X.reshape(M,1)
X_1[:10]


# In[2188]:


# Now using hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column) 
# This will be our final X matrix (feature matrix)
X = np.hstack((X_0,X_1))
X[:5]


# In[2189]:


#Making a theta array with initializations of O.
theta = np.zeros(2)
theta


# In[2190]:


def calculate_scalar(X, Y, theta): #Declaring values and computing the Scalar value J
    
    predictions = X.dot(theta)  #Dot product of array X and theta
    errors = np.subtract(predictions,Y) #Matrix subtraction with predictions and Y
    squaringErrors = np.square(errors) #Now errors contained in matrix. We square all values in matrix error.
    J = 1/(2*M)*np.sum(squaringErrors) #Scalar equation using matrix squErrors
    return J

print( 'Scalar Value is ', calculate_scalar(X, Y, theta))


# In[2191]:


# Now computing the scalar value J for the theta values
scalar_J = calculate_scalar(X, Y, theta)
print('The cost for given values of theta_0 and theta_1 =', scalar_J)


# In[2192]:


def gradient_descent(X, Y, theta, alpha, iterations):  #Function to calculate gradient descent for linear regression
    
    result = np.zeros(iterations)   #creating a row of an array with an undetermined amount of zeroes.
    theta_interval = np.zeros([iterations, theta.size])  #creating an array for each interval to be plotted (X1, X2, X3) 
    
    for i in range(iterations):    #For loop with iterations as an input.
        predictions = X.dot(theta)   #Dot product of array X and theta resulting in scalar
        errors = np.subtract(predictions,Y) #Matrix subtration between predictions and value Y
        sum_delta = (alpha/M)*X.transpose().dot(errors); #learning rate over training examples * scalar of resulting dot product.  
        theta = theta-sum_delta;   #Current theta minus scalar sum_delta for final value of theta                      
        result[i] = calculate_scalar(X, Y, theta)
        theta_interval[i] = theta #Needed to show the previous thetas used for the resulting scalar.

    return theta, result, theta_interval


# In[2193]:


#The number of iterations is kept constant to test different values for the learning rate alpha. 
#Declaring our training parameters for gradient descent for X1
theta = np.zeros(2)
iterations = 200;  #The curve seems to flatten out at around 200 iterations.
alpha_1 = 0.1;

theta, result, theta_interval = gradient_descent(X, Y, theta, alpha_1, iterations)
print('Final value of theta =', theta)
print('Y1 = ', result)


# In[2194]:


for theta in theta_interval:
    #plt.plot(X[:, 1], X.dot(theta), color = 'Blue', label = 'Linear Regression')
     plt.plot(X[:, 1], X.dot(theta))
    
#plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Explanatory Variable X1')
plt.ylabel('Y Value Y1')
plt.title('Linear Regression Fit')
#plt.legend()


# In[2195]:


#Plotting the Scalar J vs. Number of Iterations
plt.plot(range(1, iterations + 1), result, color='Red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Result for scalar J')
plt.title('Convergence for Gradient Descent Iterations')
plt.show()


# In[2196]:


#Now testing the second explanatory value X2
X = df.values[:, 1] #Getting the values of each variable in the second column
Y = df.values[:, 3] #Setting the last column as my result for y.
M = len(Y)# Number of training examples
print('X = ', X[:99])   # All variables shown given there are M = 99 samples.
print('Y = ', Y[:99])
print('M = ', M)


# In[2197]:


plt.scatter(X,Y,color='blue',marker='o')
plt.grid()
plt.rcParams["figure.figsize"]=(10,6)
plt.title('Scatter plot of training data for Homework_0')


# In[2198]:


# Creating a matrix with single column of ones
X_0 = np.ones((M,1))
X_0[:5]


# In[2199]:


# Using reshape function convert X 1D array to 2D array of dimension 2X1
X_1 = X.reshape(M,1)
X_1[:10]


# In[2200]:


# Now using hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column) 
# This will be our final X matrix (feature matrix)
X = np.hstack((X_0,X_1))
X[:5]


# In[2201]:


#Making a theta array with initializations of O.
theta = np.zeros(2)
theta


# In[2202]:


def calculate_scalar(X, Y, theta): #Declaring values and computing the Scalar value J
    
    predictions = X.dot(theta)  #Dot product of array X and theta
    errors = np.subtract(predictions,Y) #Matrix subtraction with predictions and Y
    squaringErrors = np.square(errors) #Now errors contained in matrix. We square all values in matrix error.
    J = 1/(2*M)*np.sum(squaringErrors) #Scalar equation using matrix squErrors
    return J

print( 'Scalar Value is ', calculate_scalar(X, Y, theta))


# In[2203]:


# Now computing the scalar value J for the theta values
scalar_J = calculate_scalar(X, Y, theta)
print('The cost for given values of theta_0 and theta_1 =', scalar_J)


# In[2204]:


def gradient_descent(X, Y, theta, alpha, iterations):  #Function to calculate gradient descent for linear regression
    
    result = np.zeros(iterations)   #creating a row of an array with an undetermined amount of zeroes.
    theta_interval = np.zeros([iterations, theta.size])  #creating an array for each interval to be plotted (X1, X2, X3) 
    
    for i in range(iterations):    #For loop with iterations as an input.
        predictions = X.dot(theta)   #Dot product of array X and theta resulting in scalar
        errors = np.subtract(predictions,Y) #Matrix subtration between predictions and value Y
        sum_delta = (alpha/M)*X.transpose().dot(errors); #learning rate over training examples * scalar of resulting dot product.  
        theta = theta-sum_delta;   #Current theta minus scalar sum_delta for final value of theta                      
        result[i] = calculate_scalar(X, Y, theta)
        theta_interval[i] = theta #Needed to show the previous thetas used for the resulting scalar.

    return theta, result, theta_interval


# In[2205]:


#Declaring our training parameters for gradient descent for X2
theta = np.zeros(2)
iterations = 200;  #The curve seems to flatten out at around 50 iterations.
alpha_1 = 0.01;

theta, result, theta_interval = gradient_descent(X, Y, theta, alpha_1, iterations)
print('Final value of theta =', theta)
print('Y1 = ', result)


# In[2206]:


for theta in theta_interval:
    #plt.plot(X[:, 1], X.dot(theta), color = 'Blue', label = 'Linear Regression')
     plt.plot(X[:, 1], X.dot(theta))
    
#plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Explanatory Variable X1')
plt.ylabel('Y Value Y1')
plt.title('Linear Regression Fit')
#plt.legend()


# In[2207]:


#Plotting the Scalar J vs. Number of Iterations
plt.plot(range(1, iterations + 1), result, color='Red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Result for scalar J')
plt.title('Convergence for Gradient Descent Iterations')
plt.show()


# In[2208]:


#Now testing the second explanatory value X3
X = df.values[:, 2] #Getting the values of each variable in the second column
Y = df.values[:, 3] #Setting the last column as my result for y.
M = len(Y)# Number of training examples
print('X = ', X[:99])   # All variables shown given there are M = 99 samples.
print('Y = ', Y[:99])
print('M = ', M)


# In[2209]:


plt.scatter(X,Y,color='blue',marker='o')
plt.grid()
plt.rcParams["figure.figsize"]=(10,6)
plt.title('Scatter plot of training data for Homework_0')


# In[2210]:


# Creating a matrix with single column of ones
X_0 = np.ones((M,1))
X_0[:5]


# In[2211]:


# Using reshape function convert X 1D array to 2D array of dimension 2X1
X_1 = X.reshape(M,1)
X_1[:10]


# In[2212]:


# Now using hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column) 
# This will be our final X matrix (feature matrix)
X = np.hstack((X_0,X_1))
X[:5]


# In[2213]:


#Making a theta array with initializations of O.
theta = np.zeros(2)
theta


# In[2214]:


def calculate_scalar(X, Y, theta): #Declaring values and computing the Scalar value J
    
    predictions = X.dot(theta)  #Dot product of array X and theta
    errors = np.subtract(predictions,Y) #Matrix subtraction with predictions and Y
    squaringErrors = np.square(errors) #Now errors contained in matrix. We square all values in matrix error.
    J = 1/(2*M)*np.sum(squaringErrors) #Scalar equation using matrix squErrors
    return J

print( 'Scalar Value is ', calculate_scalar(X, Y, theta))


# In[2215]:


# Now computing the scalar value J for the theta values
scalar_J = calculate_scalar(X, Y, theta)
print('The cost for given values of theta_0 and theta_1 =', scalar_J)


# In[2216]:


def gradient_descent(X, Y, theta, alpha, iterations):  #Function to calculate gradient descent for linear regression
    
    result = np.zeros(iterations)   #creating a row of an array with an undetermined amount of zeroes.
    theta_interval = np.zeros([iterations, theta.size])  #creating an array for each interval to be plotted (X1, X2, X3) 
    
    for i in range(iterations):    #For loop with iterations as an input.
        predictions = X.dot(theta)   #Dot product of array X and theta resulting in scalar
        errors = np.subtract(predictions,Y) #Matrix subtration between predictions and value Y
        sum_delta = (alpha/M)*X.transpose().dot(errors); #learning rate over training examples * scalar of resulting dot product.  
        theta = theta-sum_delta;   #Current theta minus scalar sum_delta for final value of theta                      
        result[i] = calculate_scalar(X, Y, theta)
        theta_interval[i] = theta #Needed to show the previous thetas used for the resulting scalar.

    return theta, result, theta_interval


# In[2217]:


#Declaring our training parameters for gradient descent for X3
theta = np.zeros(2)
iterations = 200;  #The curve seems to flatten out at around 150 iterations.
alpha_3 = 0.1;

theta, result, theta_interval = gradient_descent(X, Y, theta, alpha_3, iterations)
print('Final value of theta =', theta)
print('Y3 = ', result)


# In[2218]:


for theta in theta_interval:
    #plt.plot(X[:, 1], X.dot(theta), color = 'Blue', label = 'Linear Regression')
     plt.plot(X[:, 1], X.dot(theta))
    
#plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Explanatory Variable X1')
plt.ylabel('Y Value Y1')
plt.title('Linear Regression Fit')
#plt.legend()


# In[2219]:


#Plotting the Scalar J vs. Number of Iterations
plt.plot(range(1, iterations + 1), result, color='Red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Result for scalar J')
plt.title('Convergence for Gradient Descent Iterations')
plt.show()


# In[2220]:


#Code below is to work on Problem #2

#Reading the data from D3
df = pd.read_csv("D3.csv")
df.head()
M = len(df)
M


# In[2221]:


#Extracting the data into 1D matrices 

X_1 = df.values[:, 0] # get input values from the first column X_0
X_2 = df.values[:, 1] # get input values from the second column x_1
X_3 = df.values[:, 2] # get input values from the third column X_3
Y = df.values[:, 3] # get output values from the fourth column Y

M = len(Y) # Number of training examples
print('X = ', X_1[: 99]) # Show all the data points for X_1
print('X = ', X_2[: 99]) # Show all the data points for X_2
print('X = ', X_3[: 99]) # Show all the data points for X_3
print('Y = ', Y[: 99]) # Show all the data points for Y
print('M = ', M)


# In[2222]:


# Using reshape function convert all X variables 1D array to 2D array of dimension 3X1
X = np.ones((M,1))
X_1 = X_1.reshape(M,1)
X_2 = X_2.reshape(M,1)
X_3 = X_3.reshape(M,1)

X = np.hstack((X,X_1,X_2,X_3))


# In[2223]:


#Making a theta array with initializations of O and setting training parameters.
theta = np.zeros(4)
iterations = 500;
alpha = 0.1;
result = calculate_scalar(X,Y, theta)
print('Scalar values is ', result) #Print the scalar value for Gradient Descent


# In[2224]:


#Calculating gradient descent with theta and scalar J
theta, result, theta_interval = gradient_descent(X, Y, theta, alpha, iterations)
print('Final value of theta =', theta)
print('Y = ', result)


# In[2225]:


for theta in theta_interval:
    #plt.plot(X[:, 1], X.dot(theta), color = 'Blue', label = 'Linear Regression')
     plt.plot(X[:, 1], X.dot(theta))
    
#plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('All Explanatory Variables')
plt.ylabel('Y Value')
plt.title('Linear Regression Fit')
#plt.legend()


# In[2226]:


#Plotting the Scalar J vs. Number of Iterations for all X values combined
plt.plot(range(1, iterations + 1), result, color='Red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Result for scalar J')
plt.title('Convergence for Gradient Descent for All 3 Explanatory Variables')
plt.show()


# In[2227]:


def prediction(X, theta):
    return theta[0]+theta[1]*X[0]+theta[2]*X[1]*X[2]
new_X1 = np.array((1,1,1))
Y = prediction(new_X1, theta)
print(Y)

new_X2 = np.array((2,0,4))
Y = prediction(new_X2, theta)
print(Y)

new_X3 = np.array((3,2,1))
Y = prediction(new_X3, theta)
print(Y)

