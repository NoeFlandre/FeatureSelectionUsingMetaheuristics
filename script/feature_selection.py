# Import Libraries
import numpy as np
import pandas as pd
import time
import argparse
from sklearn.feature_selection import f_classif
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import random
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Disable all warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Define Genetic Algorithm Selector
class GeneticAlgorithmFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector using Genetic Algorithm.
    """
    def __init__(self, num_features, fitness_func, num_generations=10, population_size=20, mutation_rate=0.1):
        self.num_features = num_features
        self.fitness_func = fitness_func
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def fit(self, X, y=None):
        """
        Fit the genetic algorithm to select features.
        """
        self.num_total_features = X.shape[1]
        # Initialize population with exactly num_features selected
        population = [self.initialize_individual() for _ in range(self.population_size)]
        for generation in range(self.num_generations):
            print(f"  Generation {generation+1}/{self.num_generations}")
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(individual, X, y) for individual in population]
            # Selection
            selected = self.selection(population, fitness_scores)
            # Crossover
            offspring = self.crossover(selected)
            # Mutation
            population = self.mutation(offspring)
        # Get the best individual
        fitness_scores = [self.evaluate_fitness(individual, X, y) for individual in population]
        best_idx = np.argmin(fitness_scores)
        self.best_individual_ = population[best_idx]
        print(f"  Best fitness score: {fitness_scores[best_idx]:.4f}")
        return self

    def evaluate_fitness(self, individual, X, y=None):
        """
        Evaluate the fitness of an individual.
        """
        if y is not None:
            # Use supervised fitness function
            return self.fitness_func(individual, X, y)
        else:
            # Use unsupervised fitness function
            return self.fitness_func(individual, X)

    def transform(self, X):
        """
        Transform the data to select features based on the best individual.
        """
        return X[:, self.best_individual_ == 1]

    def initialize_individual(self):
        """
        Initialize an individual with exactly num_features selected.
        """
        individual = np.zeros(self.num_total_features, dtype=int)
        selected_indices = np.random.choice(self.num_total_features, self.num_features, replace=False)
        individual[selected_indices] = 1
        return individual

    def selection(self, population, fitness_scores):
        """
        Select individuals for the next generation using tournament selection.
        """
        selected = []
        for _ in range(self.population_size):
            idx1, idx2 = random.sample(range(len(population)), 2)
            if fitness_scores[idx1] < fitness_scores[idx2]:
                selected.append(population[idx1])
            else:
                selected.append(population[idx2])
        return selected

    def crossover(self, selected):
        """
        Perform crossover between pairs of individuals.
        """
        offspring = []
        for i in range(0, self.population_size, 2):
            parent1 = selected[i].copy()
            parent2 = selected[(i+1) % self.population_size].copy()
            if random.random() < 0.8:
                point = random.randint(1, self.num_total_features - 1)
                child1 = np.concatenate([parent1[:point], parent2[point:]])
                child2 = np.concatenate([parent2[:point], parent1[point:]])
            else:
                child1, child2 = parent1, parent2
            offspring.extend([child1, child2])
        return offspring

    def mutation(self, offspring):
        """
        Apply mutation to the offspring.
        """
        for individual in offspring:
            for i in range(self.num_total_features):
                if random.random() < self.mutation_rate:
                    individual[i] = 1 - individual[i]
            # Ensure exactly num_features are selected
            selected = np.sum(individual)
            if selected > self.num_features:
                excess = selected - self.num_features
                indices = np.where(individual == 1)[0]
                deselect = np.random.choice(indices, excess, replace=False)
                individual[deselect] = 0
            elif selected < self.num_features:
                deficit = self.num_features - selected
                indices = np.where(individual == 0)[0]
                select = np.random.choice(indices, deficit, replace=False)
                individual[select] = 1
        return offspring

# Define Particle Swarm Optimization Selector
class ParticleSwarmFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector using Particle Swarm Optimization.
    """
    def __init__(self, num_features, fitness_func, num_particles=20, max_iter=10, inertia=0.5, cognitive=1, social=2):
        self.num_features = num_features
        self.fitness_func = fitness_func
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

    def fit(self, X, y=None):
        """
        Fit the PSO to select features.
        """
        self.num_total_features = X.shape[1]
        # Initialize particles
        particles = [self.initialize_particle() for _ in range(self.num_particles)]
        velocities = [np.zeros(self.num_total_features) for _ in range(self.num_particles)]
        personal_best_positions = particles.copy()
        personal_best_scores = [self.evaluate_fitness(p, X, y) for p in particles]
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        for iteration in range(self.max_iter):
            print(f"  Iteration {iteration+1}/{self.max_iter}")
            for i, particle in enumerate(particles):
                # Update velocity
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.cognitive * random.random() * (personal_best_positions[i] - particle)
                    + self.social * random.random() * (global_best_position - particle)
                )
                # Update position
                particle = particle + velocities[i]
                # Apply sigmoid function to convert to probabilities
                probabilities = 1 / (1 + np.exp(-particle))
                # Update particle's position based on probabilities
                particle = (probabilities > 0.5).astype(int)
                # Ensure exactly num_features are selected
                selected = np.sum(particle)
                if selected > self.num_features:
                    excess = selected - self.num_features
                    indices = np.where(particle == 1)[0]
                    deselect = np.random.choice(indices, excess, replace=False)
                    particle[deselect] = 0
                elif selected < self.num_features:
                    deficit = self.num_features - selected
                    indices = np.where(particle == 0)[0]
                    if len(indices) >= deficit:
                        select = np.random.choice(indices, deficit, replace=False)
                        particle[select] = 1
                    else:
                        # If not enough features to select, select all
                        particle[indices] = 1
                particles[i] = particle
                # Evaluate fitness
                fitness = self.evaluate_fitness(particle, X, y)
                # Update personal best
                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = particle.copy()
            # Update global best
            global_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[global_best_idx] < global_best_score:
                global_best_position = personal_best_positions[global_best_idx].copy()
                global_best_score = personal_best_scores[global_best_idx]
        self.best_individual_ = global_best_position
        print(f"  Best fitness score: {global_best_score:.4f}")
        return self

    def evaluate_fitness(self, individual, X, y=None):
        """
        Evaluate the fitness of a particle.
        """
        if y is not None:
            return self.fitness_func(individual, X, y)
        else:
            return self.fitness_func(individual, X)

    def transform(self, X):
        """
        Transform the data to select features based on the best particle.
        """
        return X[:, self.best_individual_ == 1]

    def initialize_particle(self):
        """
        Initialize a particle with exactly num_features selected.
        """
        particle = np.zeros(self.num_total_features, dtype=float)
        selected_indices = np.random.choice(self.num_total_features, self.num_features, replace=False)
        particle[selected_indices] = 1.0
        return particle


# Define Simulated Annealing Selector
class SimulatedAnnealingFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector using Simulated Annealing.
    """
    def __init__(self, num_features, fitness_func, max_iter=1000, initial_temp=100, cooling_rate=0.99):
        self.num_features = num_features
        self.fitness_func = fitness_func
        self.max_iter = max_iter
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def fit(self, X, y=None):
        """
        Fit the Simulated Annealing algorithm to select features.
        """
        self.num_total_features = X.shape[1]
        current_solution = self.initialize_solution()
        current_fitness = self.evaluate_fitness(current_solution, X, y)
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        temp = self.initial_temp

        for iteration in range(self.max_iter):
            if temp <= 0.1:
                break
            # Generate neighbor solution
            neighbor = self.generate_neighbor(current_solution)
            neighbor_fitness = self.evaluate_fitness(neighbor, X, y)
            delta_fitness = neighbor_fitness - current_fitness
            # Decide whether to accept the neighbor
            if delta_fitness < 0 or random.random() < np.exp(-delta_fitness / temp):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                # Update best solution
                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
            # Cool down
            temp *= self.cooling_rate
        self.best_individual_ = best_solution
        print(f"  Best fitness score: {best_fitness:.4f}")
        return self

    def evaluate_fitness(self, individual, X, y=None):
        """
        Evaluate the fitness of a solution.
        """
        if y is not None:
            return self.fitness_func(individual, X, y)
        else:
            return self.fitness_func(individual, X)

    def transform(self, X):
        """
        Transform the data to select features based on the best solution.
        """
        return X[:, self.best_individual_ == 1]

    def initialize_solution(self):
        """
        Initialize a solution with exactly num_features selected.
        """
        solution = np.zeros(self.num_total_features, dtype=int)
        selected_indices = np.random.choice(self.num_total_features, self.num_features, replace=False)
        solution[selected_indices] = 1
        return solution

    def generate_neighbor(self, solution):
        """
        Generate a neighbor solution by flipping bits.
        """
        neighbor = solution.copy()
        # Flip a random feature
        flip_index = np.random.randint(0, self.num_total_features)
        neighbor[flip_index] = 1 - neighbor[flip_index]
        # Ensure exactly num_features are selected
        selected = np.sum(neighbor)
        if selected > self.num_features:
            excess = selected - self.num_features
            indices = np.where(neighbor == 1)[0]
            deselect = np.random.choice(indices, excess, replace=False)
            neighbor[deselect] = 0
        elif selected < self.num_features:
            deficit = self.num_features - selected
            indices = np.where(neighbor == 0)[0]
            select = np.random.choice(indices, deficit, replace=False)
            neighbor[select] = 1
        return neighbor

# Define Fitness Functions
def optimized_unsupervised_fitness(individual, X):
    """
    Fitness function for the unsupervised setting using mutual information.
    """
    selected_features = np.where(individual == 1)[0]
    if len(selected_features) < 2:
        return np.inf  # Need at least two features to compute mutual information
    X_selected = X[:, selected_features]
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    try:
        X_disc = discretizer.fit_transform(X_selected)
    except ValueError as e:
        print(f"    Discretizer error: {e}")
        return np.inf

    central_feature = X_disc[:, 0]
    try:
        mi_scores = Parallel(n_jobs=-1)(
            delayed(mutual_info_score)(central_feature, X_disc[:, i]) for i in range(1, X_disc.shape[1])
        )
    except Exception as e:
        print(f"    Mutual information computation error: {e}")
        return np.inf

    mi_sum = np.sum(mi_scores)
    # Aim to minimize redundancy and select desired number of features
    return mi_sum

# Inside the supervised_fitness function
def supervised_fitness(individual, X, y):
    selected_features = np.where(individual == 1)[0]
    if len(selected_features) == 0:
        return np.inf  # Penalize empty feature sets
    X_selected = X[:, selected_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=5000, solver='lbfgs', class_weight='balanced'))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    try:
        auc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc_ovr', n_jobs=-1)
        mean_auc = np.mean(auc_scores)
    except Exception as e:
        print(f"    Exception during cross-validation: {e}")
        return np.inf
    # Negative because we minimize
    return -mean_auc

# Main Execution Function
def main():
    # Argument Parser for Dataset Path
    parser = argparse.ArgumentParser(description='Feature Selection Script')
    parser.add_argument('data_path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV file for results')
    args = parser.parse_args()

    # Load Data
    data = pd.read_csv(args.data_path)

    # Data Preprocessing
    X = data.drop('Class', axis=1).values
    y = data['Class'].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    # Check class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_classes, counts))}")

    # Define Classifiers with Tuned Hyperparameters
    classifiers = {
        'Logistic Regression': OneVsRestClassifier(
            LogisticRegression(max_iter=2000, solver='saga', class_weight='balanced')
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, class_weight='balanced'
        ),
        'Support Vector Machine': SVC(
            probability=True, kernel='linear', random_state=42, class_weight='balanced'
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5, weights='distance'
        )
    }

    # Define how many features to select
    num_features_to_select = 50

    # Initialize Results Dictionary
    results = defaultdict(list)

    # List of selectors and settings
    selectors = [
        ('GA Unsupervised', GeneticAlgorithmFeatureSelector(
            num_features=num_features_to_select,
            fitness_func=optimized_unsupervised_fitness,
            num_generations=10,
            population_size=20,
            mutation_rate=0.1
        )),
        ('GA Supervised', GeneticAlgorithmFeatureSelector(
            num_features=num_features_to_select,
            fitness_func=supervised_fitness,
            num_generations=10,
            population_size=20,
            mutation_rate=0.1
        )),
        ('PSO Unsupervised', ParticleSwarmFeatureSelector(
            num_features=num_features_to_select,
            fitness_func=optimized_unsupervised_fitness,
            num_particles=20,
            max_iter=10
        )),
        ('PSO Supervised', ParticleSwarmFeatureSelector(
            num_features=num_features_to_select,
            fitness_func=supervised_fitness,
            num_particles=20,
            max_iter=10
        )),
        ('SA Unsupervised', SimulatedAnnealingFeatureSelector(
            num_features=num_features_to_select,
            fitness_func=optimized_unsupervised_fitness,
            max_iter=1000,
            initial_temp=100,
            cooling_rate=0.99
        )),
        ('SA Supervised', SimulatedAnnealingFeatureSelector(
            num_features=num_features_to_select,
            fitness_func=supervised_fitness,
            max_iter=1000,
            initial_temp=100,
            cooling_rate=0.99
        ))
    ]

    # Main Execution Loop with Comprehensive Debugging and Execution Time Tracking
    for selector_name, selector in selectors:
        print(f"\nProcessing {selector_name}...")
        start_time = time.time()
        try:
            # Fit the selector
            if "Unsupervised" in selector_name:
                selector.fit(X)  # No y for unsupervised
            else:
                selector.fit(X, y)

            X_selected = selector.transform(X)
            execution_time = time.time() - start_time

            num_selected = X_selected.shape[1]
            print(f"  {selector_name}: Selected {num_selected} features. Shape: {X_selected.shape}")

            # Check for constant features
            variances = np.var(X_selected, axis=0)
            num_constant = np.sum(variances == 0)
            if num_constant > 0:
                print(f"    {selector_name}: {num_constant} constant features detected.")
            else:
                print(f"    {selector_name}: No constant features detected.")

            # Check for missing values
            missing_values = np.isnan(X_selected).sum()
            if missing_values > 0:
                print(f"    {selector_name}: {missing_values} missing values detected in selected features.")
            else:
                print(f"    {selector_name}: No missing values detected in selected features.")

            # Scale the selected features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)

            # Evaluate classifiers
            for clf_name, clf in classifiers.items():
                try:
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    auc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc_ovr', n_jobs=-1)
                    mean_auc = np.mean(auc_scores)
                    std_auc = np.std(auc_scores)
                    results['Method'].append(selector_name)
                    results['Classifier'].append(clf_name)
                    results['AUC Mean'].append(mean_auc)
                    results['AUC Std'].append(std_auc)
                    results['Execution Time'].append(execution_time)
                    print(f"    {clf_name}: AUC Mean = {mean_auc:.4f}, AUC Std = {std_auc:.4f}")
                except Exception as e:
                    print(f"    {clf_name}: Error during evaluation: {e}")
                    results['Method'].append(selector_name)
                    results['Classifier'].append(clf_name)
                    results['AUC Mean'].append(np.nan)
                    results['AUC Std'].append(np.nan)
                    results['Execution Time'].append(execution_time)

        except Exception as e:
            print(f"  {selector_name}: Error during selection process: {e}")
            for clf_name in classifiers.keys():
                results['Method'].append(selector_name)
                results['Classifier'].append(clf_name)
                results['AUC Mean'].append(np.nan)
                results['AUC Std'].append(np.nan)
                results['Execution Time'].append(np.nan)

    # Convert Results to DataFrame
    results_df = pd.DataFrame(results)

    # Display the Results Table
    print("\nResults:")
    print(results_df)

    # Save Results to CSV
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
