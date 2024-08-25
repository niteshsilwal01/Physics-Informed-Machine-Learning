# Physics-Informed Machine Learning for Simple Harmonic Oscillator
This repository contains the implementation of a Physics-Informed Neural Network (PINN) designed to solve and simulate the behavior of a Simple Harmonic Oscillator (SHO). The SHO is a fundamental physical system characterized by periodic motion, often described by second-order differential equations. This project leverages the power of machine learning by embedding the physical laws governing the SHO directly into the neural network's architecture.

![Simple Harmonic Oscillator](https://github.com/niteshsilwal01/Physics-Informed-Machine-Learning/blob/main/oscillation_frames/Damped_oscillator.gif?raw=true)
![Simple Harmonic Oscillator](https://github.com/niteshsilwal01/Physics-Informed-Machine-Learning/blob/main/oscillation_frames/Simple%20Harmonic%20Oscillator.gif?raw=true)

## Features
### PINN Implementation: 
A neural network that incorporates the physics of the Simple Harmonic Oscillator into its loss function, ensuring the model adheres to known physical laws.
### Differential Equation Solvers: 
Uses automatic differentiation in PyTorch to compute the derivatives necessary for the physics-informed loss.
### Comparison with Analytical Solutions: 
Provides tools to compare the PINN's predictions with the exact analytical solutions of the SHO.
### Visualization: 
Includes scripts for visualizing the training process and the comparison between the neural network's predictions and the analytical solutions.
### Training and Testing: 
Scripts to train the model on collocation points and test its accuracy.

## Installation
To get started with this project, clone the repository:
[git clone https://github.com/niteshsilwal01/Damped Harmonic Oscillator.git](https://github.com/niteshsilwal01/Physics-Informed-Machine-Learning.git) and install the required dependencies: pip install -r requirements.txt

## Usage
The main script _Oscillator.py_ trains the PINN on the SHO problem. You can adjust the parameters, such as the damping coefficient, natural frequency, and neural network architecture.

## Background
Physics-Informed Neural Networks (PINNs) are a novel approach that incorporates physical laws as part of the neural network's training process. This method is particularly useful for solving differential equations where traditional data-driven models may fail to generalize.

## Contributions
Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
