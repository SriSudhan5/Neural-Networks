#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:46:47 2023

@author: ap
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
# UNCOMMENT IF USING COLAB
#from google.colab import drive
#drive.mount('/content/drive')
#IMDIR = 'add direcrory'
print(IMDIR)

def load_dataset(IMDIR):
    train_dataset = h5py.File(IMDIR+'train_catvnoncat.h5', "r")
    train_x = np.array(train_dataset["train_set_x"][:])
    train_y = np.array(train_dataset["train_set_y"][:])
    test_dataset = h5py.File(IMDIR+'test_catvnoncat.h5', "r")
    test_x = np.array(test_dataset["test_set_x"][:])
    test_y = np.array(test_dataset["test_set_y"][:])
    classes = np.array(test_dataset["list_classes"][:])
    
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42, shuffle=True)
    val_y = val_y.reshape((1, val_y.shape[0]))
    
    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x, train_y, test_x, test_y, val_x, val_y, classes

train_x, train_y, test_x, test_y, val_x, val_y, classes = load_dataset(IMDIR)

# run several times to visualize different data points
# the title shows the ground truth class labels (0=no cat , 1 = cat)
index = np.random.randint(low=0,high=train_y.shape[1])
plt.imshow(train_x[index])
plt.title("Image "+str(index)+" label "+str(train_y[0,index]))
plt.show()
print ("Train X shape: " + str(train_x.shape))
print ("We have "+str(train_x.shape[0]),
       "images of dimensionality "
       + str(train_x.shape[1])+ "x"
       + str(train_x.shape[2])+ "x"
       + str(train_x.shape[3]))


train_x, train_y, test_x, test_y, val_x, val_y, classes = load_dataset(IMDIR)
print ("Original train X shape: " + str(train_x.shape))
print ("Original test X shape: " + str(test_x.shape))
train_x = train_x.reshape(train_x.shape[0], -1).T
test_x = test_x.reshape(test_x.shape[0], -1).T
print ("Train X shape: " + str(train_x.shape))
print ("Train Y shape: " + str(train_y.shape))
print ("Test X shape: " + str(test_x.shape))
print ("Test Y shape: " + str(test_y.shape))



train_x = train_x/255.
test_x = test_x/255.
val_x = val_x/255.

def sigmoid(z):
    Z=1/(1 + np.exp(-z))
    return Z


def initialize_parameters(dim):
    w = np.random.randn(dim,1)*0.01
    b = 0
    return w, b

def neuron(w, b, x):
    pred_y = np.dot(np.transpose(w), x) + b
    return pred_y


w, b = initialize_parameters(train_x.shape[0])  # Fix the initialization of weights
pred_y = neuron(w, b, train_x)

y_pred = sigmoid(pred_y)
print("y_pred Shape: ", y_pred.shape)

def crossentropy(y,y_pred):

    m = y.shape[1]
    epsilon = 1e-12 # number of examples
    # Compute the binary cross-entropy loss
    cost = (-1/m) * np.sum(y* np.log(y_pred+epsilon) + (1 - y)* np.log(1 - y_pred+epsilon))
    return cost

cost = crossentropy(train_y, y_pred)
print("Crossentropy Loss: ", cost)

def backpropagate(X, Y, Ypred):
    m = X.shape[1]

#     #find gradient (back propagation)
#     dw =(-1/m)*np.sum(Ypred-Y)
#     db =(-1/m)*np.sum(np.dot(X,(Ypred-Y)))
    dw = (1/m) * np.matmul(X, (Ypred-Y).T)
    db = (1/m) * np.sum(Ypred-Y)

    grads = {"dw": dw,
             "db": db}

    return grads


def gradient_descent(X, Y, iterations, learning_rate):
    costs = []
    w, b = initialize_parameters(train_x.shape[0])

    for i in range(iterations):
        Ypred = sigmoid(neuron(w, b, X))
        cost = crossentropy(Y, Ypred)
        grads = backpropagate(X, Y, Ypred)

        # Update parameters
        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]
        costs.append(cost)

        if i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return w, b, costs

w, b, costs = gradient_descent(train_x, train_y, iterations=2000, learning_rate=0.005)


plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()


def predict(w, b, X):
    y_pred = sigmoid(np.dot(np.transpose(w), X) + b)
    return y_pred


# predict
train_pred_y = predict(w, b, train_x)
test_pred_y = predict(w, b, test_x)
print("Train Acc: {} %".format(100 - np.mean(np.abs(train_pred_y - train_y)) * 100))
print("Test Acc: {} %".format(100 - np.mean(np.abs(test_pred_y - test_y)) * 100))


def find_best_learning_rate(train_x, train_y, val_x, val_y, test_x, test_y, iterations, learning_rates):
    best_accuracy = 0
    best_learning_rate = None
    best_w = None 
    best_b = None
    
    for lr in learning_rates:
        print(f"Trying learning rate: {lr}")
        w, b, costs = gradient_descent(train_x, train_y, iterations, lr)
        
        # Reshape val_x to match the shape expected by the predict function
        val_x_reshaped = val_x.reshape(val_x.shape[0], -1).T
        val_pred_y = predict(w, b, val_x_reshaped)
        val_accuracy = 100 - np.mean(np.abs(val_pred_y - val_y)) * 100
        
        print(f"Validation Accuracy: {val_accuracy}%")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_learning_rate = lr
            best_w = w
            best_b = b
    
    print("\nBest Learning Rate:", best_learning_rate)
    
    # Reshape train_x and test_x to match the shape expected by the predict function
    #train_x_reshaped = train_x.reshape(train_x.shape[0], -1).T
    #test_x_reshaped = test_x.reshape(test_x.shape[0], -1).T
    
    train_pred_y = predict(best_w, best_b, train_x)
    train_accuracy = 100 - np.mean(np.abs(train_pred_y - train_y)) * 100
    print("Train Accuracy with Best Learning Rate: {}%".format(train_accuracy))
    
    test_pred_y = predict(best_w, best_b, test_x)
    test_accuracy = 100 - np.mean(np.abs(test_pred_y - test_y)) * 100
    print("Test Accuracy with Best Learning Rate: {}%".format(test_accuracy))

    return best_w, best_b

best_w, best_b = find_best_learning_rate(train_x, train_y, val_x, val_y, test_x, test_y, iterations=2000, learning_rates=[0.1, 0.01, 0.001, 0.0001])


#w, b, costs = gradient_descent(val_x, val_y, iterations=2000, learning_rate=0.005)

#print("Train Acc: {} %".format(100 - np.mean(np.abs(train_pred_y - train_y)) * 100))
#print("Test Acc: {} %".format(100 - np.mean(np.abs(test_pred_y - test_y)) * 100))
