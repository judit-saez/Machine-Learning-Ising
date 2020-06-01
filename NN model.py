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


####USEFUL FUNCTIONS####   
    
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def preprocessing(split):
    f = open("E:\\output2400_ising_classic.csv", "r")
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
        if t < Tc:
            y = 1  #ordenat
        else:
            y = 0  #desordenat
        Y.append(y)    
    f.close()    
    zipped = list(zip(X, Y, T))#[(x,y),(x,y),(x,y),...]
    
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

def crazyshuffle(arr1):
    shuffled = np.take(arr1, np.random.permutation(arr1.shape[0]),axis=0,out=arr1)
    return shuffled
   


def processing_ising_nnn_02():
    f = open(r"C:\Users\judit\Desktop\J1D\TFG\NNNI\J=1,Jd=0.2\tot_en_un.csv")
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
            y = 1  #ordenat
        else:
            y = 0  #desordenat
        Y_nnn_02.append(y)
    f.close()    
    zipped = list(zip(ising_nnn_data_02, Y_nnn_02, Temperatures))#[(x,y,T),(x,y,T),(x,y,T),...]
    
    random.shuffle(zipped)
    ising_nnn_data_02, Y_nnn_02, Temperatures = zip(*zipped)
    Y_nnn_02 = list(Y_nnn_02)
    ising_nnn_data_02 = list(ising_nnn_data_02)
    Temperatures = list(Temperatures)
    ising_nnn_data_02 = np.array(ising_nnn_data_02).T
    Y_nnn_02 = np.array(Y_nnn_02).T
    return  ising_nnn_data_02, Y_nnn_02, Temperatures

def processing_ising_nnn_s2():
    f = open(r"C:\Users\judit\Desktop\J1D\TFG\NNNI\J=1,Jd=sqrt2\tot_en_un.csv")
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
            y = 1  #ordenat
        else:
            y = 0  #desordenat
        Y_nnn_s2.append(y)
    f.close()    
    zipped = list(zip(ising_nnn_data_s2, Y_nnn_s2, Temperatures))#[(x,y,T),(x,y,T),(x,y,T),...]
    
    random.shuffle(zipped)
    ising_nnn_data_s2, Y_nnn_s2, Temperatures = zip(*zipped)
    Y_nnn_s2 = list(Y_nnn_s2)
    ising_nnn_data_s2 = list(ising_nnn_data_s2)
    Temperatures = list(Temperatures)
    ising_nnn_data_s2 = np.array(ising_nnn_data_s2).T
    Y_nnn_s2 = np.array(Y_nnn_s2).T
    return  ising_nnn_data_s2, Y_nnn_s2, Temperatures

####ESTRUCTURA#####
def layer_sizes(X,Y,nh): #X es una matriu de columnes imatge, (m imatges) (400 neurones), Y és una matriu columna de 2 neurones.
    nx = np.prod(X.shape[0]) #shape 0 = nombre de files
    nh = nh #hiperparàmetre
    ny = 1 # això hauria de donar 1
    
    return (nx, nh, ny) # retorna números

####INICIALITZACIÓ DE PARÀMETRES#####
def initialize_params(nx, nh, ny):
    w1 = np.random.randn(nh, nx)*0.1        #multipliquem per 0.01 perquè siguin prou petits
    b1 = np.zeros((nh, 1))
    w2 = np.random.randn(ny, nh)*0.1
    b2 = np.zeros((ny, 1))
    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    
    return parameters #retorna un diccionari

####ACTIVATION FUNCTIONS####
def activation(z, nl_function): #si z és un array retornarà un array
    if nl_function == "relu":
        return np.maximum(0, z)
    elif nl_function == "tanh":
        return np.tanh(z)   
    elif nl_function == "sigmoid":
        return sigmoid(z)
    
####FORWARD PROPAGATION####
def forward_propagation(X, parameters):
    z1 = np.dot(parameters["w1"], X) + parameters["b1"]
    A1 = activation(z1, "tanh")
    
    z2 = np.dot(parameters["w2"], A1) + parameters["b2"]
    A2 = activation(z2, "sigmoid")
    
    cache = {"z1": z1, "A1": A1, "z2": z2, "A2": A2}    
    return A2, cache #retorna un vector i un diccionari

####FUNCIÓ DE COST####
def compute_loss(A2, Y): 
    m = Y.shape[0]
    print(Y.shape[0])
    cost = -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) 
    cost = np.squeeze(cost)             #np.squeeze
    
    return cost 
    
####BACKWARD PROPAGATION####
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


####UPDATE PARAMETERS####
def update_params(parameters, grads, lr):
    parameters["w1"] = parameters["w1"] - lr * grads["dw1"] 
    parameters["b1"] = parameters["b1"] - lr * grads["db1"]
    parameters["w2"] = parameters["w2"] - lr * grads["dw2"]
    parameters["b2"] = parameters["b2"] - lr * grads["db2"]
    
    return parameters
    
####TRAINED NN####
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

####PREDICT####
def predict(X, parameters):
    A2, cache = forward_propagation(X, parameters)
    prediction = A2.T 
    #print(prediction.shape)
    return prediction 

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

#MAGNETIZATION DATA FOR ANALYSIS:    
MLIST_02 = [1, 1, 0.99999622, 0.999878006, 0.996227899, 0.97770909, 0.91430611,
            0.538210255, 0.184656705, 0.124115471, 0.095074775, 0.082309706, 
            0.074742677, 0.069086693]
 
TLIST_02 = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]

MLIST_S2 = [0.999993206, 0.999165381, 0.990786442, 0.964275238, 0.893491425, 
              0.815229634, 0.611970826, 0.320863011, 0.202298215, 0.123484107, 
              0.098082222, 0.085762532, 0.076485906]

TLIST_S2 = [1.5, 2.5, 3.5, 4.5, 5.5, 6.0, 6.5, 7.0, 7.5, 8.5, 9.5, 10.5, 11.5]


#TRAIN
print('TRAINING')
now = datetime.datetime.now()
Xtrain, Xtest, Ytrain, Ytest, Ttrain, Ttest = preprocessing(80)

Xtest = crazyshuffle(Xtest)

(nx, nh, ny) = layer_sizes(Xtrain, Ytrain, 300)
parameters = initialize_params(nx, nh, ny)
parameters, cost = nn_model(Xtrain, Ytrain, 15000)
#print(cost)
prediction_train = predict(Xtrain, parameters)
acc_train = (accuracy(prediction_train, Ytrain))
print("Acc_train: " + str(acc_train) + '\n')

#VALIDATION
print('VALIDATION')
prediction_test = predict(Xtest, parameters)
acc_val = accuracy(prediction_test, Ytest)
print("Acc_val: " + str(acc_val) + '\n')    


"""df = pd.DataFrame(list(zip(Ttest, prediction_test)), columns =['T', 'values xn'])
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
dff = pd.DataFrame(list(zip(T, av_pred_val)), columns =['T', '<values xn>'])"""

#TESTING
print('TESTING SQRT2')
ising_nnn_data_s2, Y_nnn_s2, Temperatures = processing_ising_nnn_s2()
ising_nnn_data_s2 = crazyshuffle(ising_nnn_data_s2)
prediction_nnn_s2 = predict(ising_nnn_data_s2, parameters)
acc_s2 = accuracy(prediction_nnn_s2, Y_nnn_s2)
print("Acc_s2: " + str(acc_s2)+ '\n') 

"""plt.scatter(Temperatures, prediction_nnn_s2, s=25, alpha=0.025, c='black', marker='o', label='output xn')
plt.scatter(TLIST_S2, MLIST_S2, c='b', marker='x', label='anàlisis mag')
plt.legend(loc="best")
plt.xlabel("Temperatura (J/$k_B$)")
plt.ylabel("Output xarxa - Magnetització ")
plt.savefig('C:\\Users\\judit\\Desktop\\J1D\\TFG\\comparació\\comp_m_nn_s2.png', dpi=1000)
plt.show()


df = pd.DataFrame(list(zip(Temperatures, prediction_nnn_s2)), columns =['T', 'values xn'])
av_pred_s2 = []
inc_pred_s2 = []
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
  
print('\n')

"""
print('TESTING 02')
ising_nnn_data_02, Y_nnn_02, Temperatures = processing_ising_nnn_02()
ising_nnn_data_02 = crazyshuffle(ising_nnn_data_02)
prediction_nnn_02 = predict(ising_nnn_data_02, parameters)
acc_02 = accuracy(prediction_nnn_02, Y_nnn_02)
print("Acc_02: " + str(acc_02)) 

"""plt.scatter(Temperatures, prediction_nnn_02, s=25, alpha=0.025, c='black', marker='o', label='output xn')
plt.scatter(TLIST_02, MLIST_02, c='b', marker='x', label='anàlisis mag')
plt.legend(loc="best")
plt.xlabel("Temperatura (J/$k_B$)")
plt.ylabel("Output xarxa - Magnetització ")
plt.savefig('C:\\Users\\judit\\Desktop\\J1D\\TFG\\comparació\\comp_m_nn_02.png', dpi=1000)
plt.show()


df = pd.DataFrame(list(zip(Temperatures, prediction_nnn_02)), columns =['T', 'values xn'])
av_pred_02 = []
inc_pred_02 = []
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
"""    
now2 = datetime.datetime.now() 
print(now2-now)    
    
"""dic_val = {"Av_pred(validació)": av_pred_val, 
       "Inc(pred_validació)": inc_pred_val}
dic_I02 = {"Av_pred(Ising_02)": av_pred_02, 
       "Inc(pred_Ising_02)": inc_pred_02}
dic_Is2 ={"Av_pred(Ising_s2)": av_pred_s2, 
       "Inc(pred_Ising_s2)": inc_pred_s2,}
res_ising = pd.DataFrame(dic_val)
res_ising_02 = pd.DataFrame(dic_I02)
res_ising_s2 = pd.DataFrame(dic_Is2)

    
res_ising.to_csv("C:\\Users\\judit\\Desktop\\J1D\\TFG\\Anàlisis resultats\\Pred_Ising.csv", decimal=',', sep=';') 
res_ising_02.to_csv("C:\\Users\\judit\\Desktop\\J1D\\TFG\\Anàlisis resultats\\Pred_Ising_02.csv", decimal=',', sep=';')
res_ising_s2.to_csv("C:\\Users\\judit\\Desktop\\J1D\\TFG\\Anàlisis resultats\\Pred_Ising_s2.csv", decimal=',', sep=';')  

print(ising_nnn_data_02)"""

image = np.ones((20,20),dtype=int)
image[1::2,::2] = -1
image[::2,1::2] = -1
#☼print(image)
#print('')
sorted_im = np.sort(image)
#print(sorted_im)

vector = sorted_im.reshape(-1, order='C')
col = np.asmatrix(vector).T



prediction_black_white = predict(col, parameters)
print('the prediction of state 50/50 with vertical boundary is: ' + str(prediction_black_white))















