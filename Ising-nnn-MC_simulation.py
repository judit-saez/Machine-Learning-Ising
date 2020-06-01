# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:03:59 2020

@author: judit
"""


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import zip_longest
import csv
import os
import datetime
import random

np.set_printoptions(linewidth = int(1e10))

###############################################################################
##################### FUNCTIONS ###############################################
###############################################################################


#Periodic Boundary conditions:
def ccp(i):
    if i > Nsites-1:
        return 0
    if i < 0:
        return Nsites-1
    else:
        return i


# Probability    
def get_probability(energy1, energy2, temperature):
    
  return np.exp((energy1 - energy2) / temperature)


#Get local energy:
def get_lenergy(N, M): 
  
    interaction1 = 1                #interaction for nearest neighbors
    interaction2 = 0.2              #interaction for next-to-nearest neighbors
    return -1*interaction1*spins[N,M]*(
                spins[ccp(N-1), M] 
                + spins[ccp(N+1), M] 
                + spins[N, ccp(M-1)] 
                + spins[N, ccp(M+1)]
                )-1*interaction2*spins[N,M]*(
                spins[ccp(N-1), ccp(M-1)] 
                + spins[ccp(N+1), ccp(M-1)] 
                + spins[ccp(N+1), ccp(M+1)] 
                + spins[ccp(N-1), ccp(M+1)]
                )
                    
                    
# Get total energy of the state:                    
def get_energy(spins):
    energy = 0
    for N in range(0, Nsites):
        for M in range(0, Nsites):
            lenergy = get_lenergy(N, M)
            energy = energy + lenergy
    return energy    

# Update the state of the system:
def update(spins, temperature):
  spins_new = np.copy(spins)
  i = np.random.randint(spins.shape[0])
  j = np.random.randint(spins.shape[1])
  spins_new[i, j] *= int(-1)
  
  deltaE = -2*get_lenergy(i,j)
    
  current_energy = get_energy(spins)
  new_energy = current_energy + deltaE
  if new_energy - current_energy < 0:
     return spins_new 
  elif get_probability(current_energy, new_energy, temperature) > np.random.random():
    return spins_new
  else:
    return spins

# Timesteps in function of T:
def Timesteps(temperature, Nsites):        
    
    if temperature >= 0.25 and temperature <= 1:
        return int(5e3)
    
    elif temperature == 1.5:
        return int(1e4) 
    
    elif temperature == 2:
        return int(2.5e4)
    
    elif temperature == 2.5:
        return int(5e4)
    
    elif temperature == 3:
        return int(5e5)
    
    elif temperature == 3.5:
        return int(2e5)
    
    elif temperature == 4:
        return int(1e5)
    
    elif temperature == 4.5 or temperature == 5 or temperature == 5.5 or temperature ==6:
        return int(5e4)
        
# Data which is used to compute average magnetization in function of T
def udata(temperature):
    if temperature >= 0.25 and temperature <= 1:
        return int(1e3)
    
    elif temperature == 1.5:
        return int(2e3) 
          
    elif temperature == 2:
        return int(1e4)
    
    elif temperature == 2.5:
        return int(2e4)
    
    elif temperature == 3:
        return int(1e5)
    
    elif temperature == 3.5:
        return int(5e4)
    
    elif temperature == 4:
        return int(4e4)
    
    elif temperature == 4.5 or temperature == 5 or temperature == 5.5 or temperature ==6:
        return int(2e4)
    
###############################################################################
######################### MAIN ################################################
###############################################################################
    
Nsites = 20                                     # system size
shape = (Nsites, Nsites)
TL=[]                                           #temperature list                                                                  
for k in np.arange(0.25, 0.8, 0.25):             
    TL.append(float(format(k,'.2f')))
for k in np.arange(1, 6.5, 0.5):             
    TL.append(float(format(k,'.2f')))


ti = datetime.datetime.now()

for i in range(0, 200): 
    os.mkdir('path\\Ising_nnn\\_' + str(i) + '_')
    LMaver = []                                 #List of average magnetization values
    OList = []                                  #output list 
    for temperature in TL:                      #initial state in function of T
        Tc=3.0
        if temperature >= Tc:               
            spins = np.random.choice([-1, 1], size=shape)
        if temperature < Tc:
            x = random.uniform(0, 1)
            if x >= 0 and x < 0.5:
                spins = -1*np.ones(shape)
            else:
                spins = np.ones(shape)
                
                
        # MonteCarlo Metropolis-Hastings          
        Mlist = []        
        timesteps = Timesteps(temperature, Nsites)
        for epoch in tqdm(range(timesteps)):       
            spins = update(spins, temperature)
            mag = np.abs(np.sum(spins)/Nsites**2)
            Mlist.append(mag)
        
        
        # Output
        noutput = spins.astype(int)      
        output = noutput.reshape(-1, order='C')     
        OList.append(output)
        
        
        # Magnetization        
        Maver = np.mean(Mlist[udata(temperature):])
        print('Average Magnetization: '+ str("{0:.5f}".format(Maver)))
        print('Temperature: ' + str(temperature))
        LMaver.append(Maver)
       
        # Plotting M vs timesteps    
        plt.plot(Mlist, linestyle='none', marker='.', c='black')
        plt.ylim(0, 1.1)
        plt.xlabel('Timesteps')
        plt.ylabel('Magnetization [$\mu$]')
        figname = "(" + str(i) + ") " + "Mvst (T=" + str (temperature) + ") " + str(Nsites) + "grid.png"
        plt.savefig('path\\Ising_nnn\\_' + str(i) + '_\\'+ str(figname), dpi=1000)
        plt.show()
        
        
    # Plotting M vs temperature
    plt.plot(TL, LMaver, linestyle='none', marker='.', c='black')
    plt.xlim(0, 7)
    plt.ylim(0, 1.1)
    plt.xlabel('Temperature [$J/k_B$]')
    plt.ylabel('<Magnetization>[$\mu$]')
    plt.savefig('path\\Ising_nnn\\_' + str(i) + '_\\'+ "(" + str(i) + ")" + 'MvsT.png', dpi=1000)
    plt.show()

    # Creating a csv and writing the output 
    d = [TL, OList, LMaver]
    export_data = zip_longest(*d, fillvalue = '')
    with open('path\\Ising_nnn\\_' + str(i) + '_\\'+ "(" + str(i) + ") " + 
              "Output("+ str(i) + ")_"+ str(Nsites) + "grid.csv", 'w', 
              encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("Temperature", "Output", "Av. Magnetization"))
        wr.writerows(export_data)
        myfile.close()

# Duration of the entire simulation
tf = datetime.datetime.now()
print(tf-ti) 

