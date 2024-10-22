# Feature Selection Using Evolutionary Algorithms

This project implements feature selection using various evolutionary algorithms: Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and Simulated Annealing (SA). It aims to help improve classification performance by selecting the most relevant features from a dataset. Both supervised and unsupervised fitness functions are included.

## Project Overview

The implemented algorithms are:

1. **Genetic Algorithm (GA)**: Selects features through a population-based search, evolving generations through selection, crossover, and mutation.
2. **Particle Swarm Optimization (PSO)**: Optimizes feature selection by simulating particles' movements in a search space.
3. **Simulated Annealing (SA)**: Optimizes feature selection by probabilistically accepting worse solutions to escape local minima, gradually cooling down to focus on better solutions.

The fitness functions can be either:
- **Supervised**: Assesses features based on classification performance.
- **Unsupervised**: Assesses features based on mutual information to reduce redundancy.

## Setup Instructions

### Prerequisites

To run the project, you need Python 3.x and the following libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
