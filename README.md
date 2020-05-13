# MLPhasesMatter
This repository contains all the scripts I have used to obtain the results exposed in my final project of Physics.
The main programs are: 
  * Ising-MC_simulation.py
  * Ising-nnn-MC-simulation.py
  * Neural-network.py
  
I used the first one to obtain the final states for a different temperatures in 20x20 Ising system through Metropolis-Hastings algorithm.

The second one uses the same algorithm but the system has next-to-nearest neighbor interactions instead of nearest neighbor.

Third one is a manual programmed neural network (just one hidden layer) which is trained with Ising nearest neighbor data and tested with the data obtained by the second program.
