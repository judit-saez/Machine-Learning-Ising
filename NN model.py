# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:50:23 2019

@author: judit
"""


import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import datetime 
import pandas as pd
import statistics

##############################################################################
######################## MATHEMATICAL FUNCTIONS ##############################
##############################################################################   
    
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))



##############################################################################
############################ DATA PROCESSING #################################
##############################################################################
    
def preprocessing(split):
    f = open("path_to_csv_file\file.csv", "r")
    f.readline()
    X = []
    Y = []
    T = []
    for line in f:
        Tc = 2.26
        line = line.split(";")
        line.pop()
        t = float(line.pop(0))
        T.append(t)
        line = [item.replace('[','') for item in line]
        line = [item.replace(']','') for item in line]
        line = line[0].split()
        for i in range(len(line)):
            line[i] = int(line[i])
        X.append(line)
        if t < Tc:                      #labelling
            y = 1  
        else:
            y = 0  
        Y.append(y)    
    f.close()    
    zipped = list(zip(X, Y, T))
    
    random.shuffle(zipped)
    X, Y, T = zip(*zipped)
    Y = list(Y)
    X = list(X)
    T = list(T)
    real_split = int(split/100*len(X))
    
    Xtrain = np.array(X[:real_split]).T
    Xtest = np.array(X[real_split:]).T
    Ytrain = np.array(Y[:real_split]).T
    Ytest = np.array(Y[real_split:]).T
    Ttrain = np.array(T[:real_split]).T
    Ttest = np.array(T[real_split:]).T
    
    return Xtrain, Xtest, Ytrain, Ytest, Ttrain, Ttest


def processing_ising_nnn_02():
    f = open(r"path_to_csv_file\file.csv")
    f.readline()
    ising_nnn_data_02 = []
    Y_nnn_02 = []
    Temperatures  = []
    for line in f:
        Tc = 3
        line = line.split(";")
        line.pop()
        t = float(line.pop(0))
        Temperatures.append(t)
        line = [item.replace('[','') for item in line]
        line = [item.replace(']','') for item in line]
        line = line[0].split()
        for i in range(len(line)):
            line[i] = int(line[i])
        ising_nnn_data_02.append(line)
        if t < Tc:      
            y = 1  
        else:
            y = 0  
        Y_nnn_02.append(y)
    f.close()    
    zipped = list(zip(ising_nnn_data_02, Y_nnn_02, Temperatures))
    
    random.shuffle(zipped)
    ising_nnn_data_02, Y_nnn_02, Temperatures = zip(*zipped)
    Y_nnn_02 = list(Y_nnn_02)
    ising_nnn_data_02 = list(ising_nnn_data_02)
    Temperatures = list(Temperatures)
    ising_nnn_data_02 = np.array(ising_nnn_data_02).T
    Y_nnn_02 = np.array(Y_nnn_02).T
    
    return  ising_nnn_data_02, Y_nnn_02, Temperatures

def processing_ising_nnn_s2():
    f = open(r"path_to_csv_file\file.csv")
    f.readline()
    ising_nnn_data_s2 = []
    Y_nnn_s2 = []
    Temperatures  = []
    for line in f:
        Tc = 6.5
        line = line.split(";")
        line.pop()
        t = float(line.pop(0))
        Temperatures.append(t)
        line = [item.replace('[','') for item in line]
        line = [item.replace(']','') for item in line]
        line = line[0].split()
        for i in range(len(line)):
            line[i] = int(line[i])
        ising_nnn_data_s2.append(line)
        if t < Tc:      
            y = 1  
        else:
            y = 0  
        Y_nnn_s2.append(y)
    f.close()    
    zipped = list(zip(ising_nnn_data_s2, Y_nnn_s2, Temperatures))
    
    random.shuffle(zipped)
    ising_nnn_data_s2, Y_nnn_s2, Temperatures = zip(*zipped)
    Y_nnn_s2 = list(Y_nnn_s2)
    ising_nnn_data_s2 = list(ising_nnn_data_s2)
    Temperatures = list(Temperatures)
    ising_nnn_data_s2 = np.array(ising_nnn_data_s2).T
    Y_nnn_s2 = np.array(Y_nnn_s2).T
    
    return  ising_nnn_data_s2, Y_nnn_s2, Temperatures



##############################################################################
############################ NEURAL NETWORK ##################################
##############################################################################


#### NN STRUCTURE: ####
def layer_sizes(X, nh):                     # Our input is a matrix of 400 component vectors since 
    nx = np.prod(X.shape[0])                # our data comes from Ising-MC_simulation.
    nh = nh                                 # Hyperparameter
    ny = 1 
    
    return (nx, nh, ny)


#### INITIALIZE WEIGHTS AND BIASES #####
def initialize_params(nx, nh, ny):
    
    w1 = np.random.randn(nh, nx)*0.1         
    b1 = np.zeros((nh, 1))
    
    w2 = np.random.randn(ny, nh)*0.1
    b2 = np.zeros((ny, 1))
    
    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    
    return parameters 


#### ACTIVATION FUNCTIONS ####
def activation(z, nl_function): 
    
    if nl_function == "tanh":
        return np.tanh(z)   
    
    elif nl_function == "sigmoid":
        return sigmoid(z)
    
    
#### FORWARD PROPAGATION ####
def forward_propagation(X, parameters):
    
    z1 = np.dot(parameters["w1"], X) + parameters["b1"]
    A1 = activation(z1, "tanh")
    
    z2 = np.dot(parameters["w2"], A1) + parameters["b2"]
    A2 = activation(z2, "sigmoid")
    
    cache = {"z1": z1, "A1": A1, "z2": z2, "A2": A2}    
    return A2, cache 


#### COST FUNCTION: Cross Entropy loss ####
def compute_loss(A2, Y): 
    m = Y.shape[0]
    print(Y.shape[0])
    cost = -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) 
    cost = np.squeeze(cost)             #np.squeeze
    
    return cost 
    

#### BACKWARD-PROPAGATION ####
def backpropagation(parameters, cache, X, Y):
    m = X.shape[1]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    w2 = parameters["w2"]
    
    dz2 = A2 - Y 
    dw2 = 1/m * np.dot(dz2, A1.T)
    db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)
    
    dz1 = np.dot(w2.T, dz2)*(1-np.power(A1, 2))
    dw1 = 1/m * np.dot(dz1, X.T) 
    db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)

    grads = {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}
    
    return grads #retorna un diccionari


#### UPDATE PARAMETERS ####
def update_params(parameters, grads, lr):
    parameters["w1"] = parameters["w1"] - lr * grads["dw1"] 
    parameters["b1"] = parameters["b1"] - lr * grads["db1"]
    parameters["w2"] = parameters["w2"] - lr * grads["dw2"]
    parameters["b2"] = parameters["b2"] - lr * grads["db2"]
    
    return parameters
    

#### GETTING A TRAINED MODEL ####
def nn_model(X, Y, epochs):
    nx = X.shape[0]
    ny = 1
    lr = 0.005
    parameters = initialize_params(nx, nh, ny)
    for i in range(epochs):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_loss(A2, Y)
        #print(cost)
        grads = backpropagation(parameters, cache, X, Y)
        parameters = update_params(parameters, grads, lr)
    
    return parameters, cost


#### PREDICT ####
def predict(X, parameters):
    A2, cache = forward_propagation(X, parameters)
    prediction = A2.T 
    #print(prediction.shape)
    return prediction 


#### ACCURACY ####
def accuracy(prediction, Ytest):
    counter = 0
    prediction.tolist()
    Ytest.tolist()
    prediction = np.round(prediction)
    for i in range (0,len(prediction)):
        if prediction[i] == Ytest[i]:
            counter = counter + 1
    true = counter/len(Ytest)*100
    string = str(counter)+"/"+str(len(Ytest))
    return true, string
 

 
ti = datetime.datetime.now()

#### GETTING ACCURACY AND COLLECTING DATA FOR THE ANALYSIS ####

# TRAINING
print('TRAINING: Using 80% Ising model data')
Xtrain, Xtest, Ytrain, Ytest, Ttrain, Ttest = preprocessing(80)
(nx, nh, ny) = layer_sizes(Xtrain, 300)
parameters = initialize_params(nx, nh, ny)
parameters, cost = nn_model(Xtrain, Ytrain, 15000)
prediction_train = predict(Xtrain, parameters)
acc_train = (accuracy(prediction_train, Ytrain))
print("Acc_train: " + str(acc_train) + '\n')

# VALIDATION
print('VALIDATION: Using 20% Ising model data')
prediction_test = predict(Xtest, parameters)
acc_val = accuracy(prediction_test, Ytest)
print("Acc_val: " + str(acc_val) + '\n')    

    # COLLECTING DATA
df = pd.DataFrame(list(zip(Ttest, prediction_test)), columns =['T', 'values xn'])
av_pred_val = []
T = []
Tlist =  [0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5]
inc_pred_val = []
for t in Tlist:
    dt = df.loc[df['T'] == t]
    data = [l.tolist() for l in dt["values xn"]]
    data = [val for sublist in data for val in sublist]
    mean = dt["values xn"].mean()
    mean = [l.tolist() for l in mean]
    inc_mean = statistics.stdev(data)
    av_pred_val.append(mean)
    inc_pred_val.append(inc_mean)
    T.append(t)
av_pred_val  = [val for sublist in av_pred_val for val in sublist]


# TESTING MODEL J_1=1, J_2 = SQRT2 
print('TESTING MODEL: J_1=1, J_2 = SQRT2')
ising_nnn_data_s2, Y_nnn_s2, Temperatures = processing_ising_nnn_s2()
prediction_nnn_s2 = predict(ising_nnn_data_s2, parameters)
acc_s2 = accuracy(prediction_nnn_s2, Y_nnn_s2)
print("Acc_s2: " + str(acc_s2)+ '\n') 

    # COLLECTING DATA
df = pd.DataFrame(list(zip(Temperatures, prediction_nnn_s2)), columns =['T', 'values xn'])
av_pred_s2 = []
inc_pred_s2 = []
TLIST_S2 = [1.5, 2.5, 3.5, 4.5, 5.5, 6.0, 6.5, 7.0, 7.5, 8.5, 9.5, 10.5, 11.5]
for t in TLIST_S2:
    dt = df.loc[df['T'] == t]
    data = [l.tolist() for l in dt["values xn"]]
    data = [val for sublist in data for val in sublist]
    mean = dt["values xn"].mean()
    inc_mean = statistics.stdev(data)
    av_pred_s2.append(mean)
    inc_pred_s2.append(inc_mean)
av_pred_s2 = [l.tolist() for l in av_pred_s2]
av_pred_s2  = [val for sublist in av_pred_s2 for val in sublist]


# TESTING MODEL J_1=1, J_2 = 0.2    
print('TESTING MODEL: J_1=1, J_2 = 0.2')
ising_nnn_data_02, Y_nnn_02, Temperatures = processing_ising_nnn_02()
prediction_nnn_02 = predict(ising_nnn_data_02, parameters)
acc_02 = accuracy(prediction_nnn_02, Y_nnn_02)
print("Acc_02: " + str(acc_02)) 

    # COLLECTING DATA
df = pd.DataFrame(list(zip(Temperatures, prediction_nnn_02)), columns =['T', 'values xn'])
av_pred_02 = []
inc_pred_02 = []
TLIST_02 = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
for t in TLIST_02:
    dt = df.loc[df['T'] == t]
    data = [l.tolist() for l in dt["values xn"]]
    data = [val for sublist in data for val in sublist]
    mean = dt["values xn"].mean()
    inc_mean = statistics.stdev(data)
    av_pred_02.append(mean)
    inc_pred_02.append(inc_mean)
av_pred_02 = [l.tolist() for l in av_pred_02]
av_pred_02 = [val for sublist in av_pred_02 for val in sublist]  
   
    
# SAVING DATA FOR THE ANALYSIS 
dic_val = {"Av_pred(validació)": av_pred_val, 
       "Inc(pred_validació)": inc_pred_val}
dic_I02 = {"Av_pred(Ising_02)": av_pred_02, 
       "Inc(pred_Ising_02)": inc_pred_02}
dic_Is2 ={"Av_pred(Ising_s2)": av_pred_s2, 
       "Inc(pred_Ising_s2)": inc_pred_s2,}
res_ising = pd.DataFrame(dic_val)
res_ising_02 = pd.DataFrame(dic_I02)
res_ising_s2 = pd.DataFrame(dic_Is2)


# EXPOPRTING DATA FOR THE ANALYSIS 
res_ising.to_csv("C:\\Users\\judit\\Desktop\\J1D\\TFG\\Anàlisis resultats\\Pred_Ising.csv", decimal=',', sep=';') 
res_ising_02.to_csv("C:\\Users\\judit\\Desktop\\J1D\\TFG\\Anàlisis resultats\\Pred_Ising_02.csv", decimal=',', sep=';')
res_ising_s2.to_csv("C:\\Users\\judit\\Desktop\\J1D\\TFG\\Anàlisis resultats\\Pred_Ising_s2.csv", decimal=',', sep=';')  

# ENTIRE TIME OF THE SIMULATION
tf = datetime.datetime.now() 
print(tf-ti)

















