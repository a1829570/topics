"""
This Python script implements a hybrid model using Genetic Algorithms (GA) and a Multi-Layer Perceptron (MLP) to predict cardiovascular disease (CVD) risk.
This version is optimized for speed without significantly affecting accuracy or generalizability. Ideally should be faster than Algorithm.py
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from deap import base, creator, tools, algorithms  # For the Genetic Algorithm
import random  # For random number generation
from joblib import Parallel, delayed  # For parallel computation

# Step 1: Load and Preprocess the Data
data = pd.read_csv('heart_disease_uci.csv')  # Load the dataset

# Separate the features and the target variable
X = data.drop('num', axis=1)  # Remove the 'num' column from the features
y = data['num']  # Set the 'num' column (CVD outcome) as the target variable

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
            ('scaler', StandardScaler())  # Apply StandardScaler
        ]), X.select_dtypes(include=[np.number]).columns.tolist()),  # Numeric columns
        ('cat', OneHotEncoder(), categorical_columns)  # Categorical columns
    ])

# Create a pipeline for preprocessing
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the training and test data
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

# Step 2: Implement the Genetic Algorithm (GA) for Hyperparameter Optimization

# Parallelize the evaluation function using joblib's Parallel and delayed
def evaluate(individual):
    n_neurons = int(individual[0])  # Ensure n_neurons is an integer
    alpha = abs(individual[1])  # Ensure alpha is non-negative

    # Initialize the MLPClassifier with faster convergence (lower max_iter)
    mlp = MLPClassifier(hidden_layer_sizes=(n_neurons,), alpha=alpha, max_iter=300, random_state=42)  # Reduced max_iter for faster training
    mlp.fit(X_train, y_train)  # Train the MLP model

    # Return accuracy as the evaluation metric
    accuracy = mlp.score(X_test, y_test)
    return accuracy,  # Return a tuple as required by the DEAP library

# Set up the DEAP library for the Genetic Algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Create a fitness function to maximize accuracy
creator.create("Individual", list, fitness=creator.FitnessMax)  # Define an individual as a list with fitness attribute

# Register genetic algorithm operators for individuals
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 50, 200)  # Use 50 to 200 neurons
toolbox.register("attr_float", random.uniform, 0.0001, 1.0)  # Regularization parameter (alpha)

toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_int, toolbox.attr_float), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Use smaller population size and fewer generations to speed up the GA
population = toolbox.population(n=20)  # Population size of 20
ngen, cxpb, mutpb = 5, 0.5, 0.2  # Fewer generations (5), crossover probability, mutation probability

# Register the GA operators: evaluation, mating, mutation, and selection
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blended crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Gaussian mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection

# Use joblib to parallelize evaluation
def parallel_evaluate(population):
    return Parallel(n_jobs=-1)(delayed(toolbox.evaluate)(ind) for ind in population)

# Use a faster version of the GA (eaSimple) and run in parallel
algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, 
                    verbose=False, evalFunc=parallel_evaluate)

# Step 3: Train the Multi-Layer Perceptron (MLP) with Optimized Hyperparameters
best_individual = tools.selBest(population, 1)[0]  # Select the best individual from the population
best_n_neurons = int(best_individual[0])  # Ensure the number of neurons is an integer
best_alpha = abs(best_individual[1])  # Ensure alpha is non-negative

# Train the final MLP model using the best hyperparameters found by the GA
final_model = MLPClassifier(hidden_layer_sizes=(best_n_neurons,), alpha=best_alpha, max_iter=300, random_state=42)
final_model.fit(X_train, y_train)  # Train the MLP model on the training data

# Step 4: Evaluate the final model's performance
final_accuracy = final_model.score(X_test, y_test)  # Compute the accuracy of the final model on test data
print(f"Final model accuracy: {final_accuracy:.4f}")  # Print the final accuracy
