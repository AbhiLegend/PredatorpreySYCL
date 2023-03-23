# PredatorpreySYCL
<br>The problem statement solved by the given code is to simulate the population dynamics of a predator-prey ecosystem using the Lotka-Volterra equations. The Lotka-Volterra equations are a pair of first-order nonlinear differential equations that describe the interactions between two species in a closed ecosystem, where one species (the predator) feeds on the other (the prey).</br>

<br>The code initializes the initial populations of prey and predator and creates SYCL buffers to store the populations. Then, it submits a SYCL kernel using a parallel_for loop that calculates the populations at each time step using the Lotka-Volterra equations. The SYCL kernel ensures that the populations remain non-negative. After the simulation is complete, the final populations of prey and predator are printed to the console.

The Lotka-Volterra equations used in the code are:

dP/dt = aP - bPQ
dQ/dt = -cQ + dPQ

where P is the population of prey, Q is the population of predator, a is the natural growth rate of prey in the absence of predation, b is the rate at which predators consume prey, c is the natural death rate of predators in the absence of food, and d is the efficiency of turning prey into predator offspring.</br>
