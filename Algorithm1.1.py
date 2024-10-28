"""
This Python script implements a hybrid model using Genetic Algorithms (GA) and a Multi-Layer Perceptron (MLP) to predict cardiovascular disease (CVD) risk.
It also includes enhancements to address complexity and overfitting by incorporating regularization techniques, cross-validation during GA optimization, and model simplification.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score  # Added cross_val_score for k-fold cross-validation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier  # To implement the Multi-Layer Perceptron model
from deap import base, creator, tools, algorithms  # To implement the Genetic Algorithm
import random  # For random number generation, used by the GA
import shap  # Import SHAP for interpretability analysis

# Step 1: Load and Preprocess the Data
data = pd.read_csv('heart_disease_uci.csv')  # Load the dataset from the file

# Separate the features and the target variable
X = data.drop('num', axis=1)  # Remove the 'num' column from the features
y = data['num']  # Set the 'num' column (CVD outcome) as the target variable

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

# Create a ColumnTransformer to apply OneHotEncoding to categorical columns and SimpleImputer/StandardScaler to numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
            ('scaler', StandardScaler())  # Apply StandardScaler after imputation
        ]), X.select_dtypes(include=[np.number]).columns.tolist()),  # Numeric columns
        ('cat', OneHotEncoder(), categorical_columns)  # Categorical columns
    ])

# Create a pipeline to streamline preprocessing
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Ensure reproducibility with random_state

# Fit the pipeline on the training data and transform both training and test data
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

# Step 2: Implement the Genetic Algorithm (GA) for Hyperparameter Optimization

# Register the Genetic Algorithm with simplified model architecture and regularization
def evaluate(individual):
    n_neurons = int(individual[0])  # Ensure n_neurons is an integer
    alpha = abs(individual[1])  # Ensure alpha is non-negative
    
    # Initialize the MLPClassifier with the current hyperparameters and stronger regularization
    # Alpha now controls the L2 regularization strength, increased to address overfitting
    mlp = MLPClassifier(hidden_layer_sizes=(n_neurons,), alpha=alpha, max_iter=2000, random_state=42)

    # Cross-Validation: Perform 5-fold cross-validation to get a more reliable estimate of the model's performance
    # This step ensures the model's performance generalizes well across different subsets of the data
    scores = cross_val_score(mlp, X_train, y_train, cv=5)

    # Return the mean accuracy score across the folds as the evaluation metric
    return np.mean(scores),  # Return a tuple as required by the DEAP library

# Set up the DEAP library for the Genetic Algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Create a fitness function to maximize accuracy
creator.create("Individual", list, fitness=creator.FitnessMax)  # Define an individual as a list with fitness attribute

# Register genetic algorithm operators for individuals
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 50, 100)  # Simplifying the model by reducing the number of neurons (50 to 100)
# This reduces the complexity of the MLP, making it less prone to overfitting
toolbox.register("attr_float", random.uniform, 0.001, 1.0)  # Register a float attribute for the alpha parameter (0.001 to 1.0)
# We are increasing the alpha range slightly to strengthen regularization and combat overfitting

toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_int, toolbox.attr_float), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the GA operators: evaluation, mating, mutation, and selection
toolbox.register("evaluate", evaluate)  # Register the evaluation function
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Register the crossover operator with a blending method
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Register the mutation operator with Gaussian distribution
toolbox.register("select", tools.selTournament, tournsize=3)  # Register the selection operator with tournament size 3

# Initialize and run the Genetic Algorithm
population = toolbox.population(n=20)  # Create a population of 20 individuals
ngen, cxpb, mutpb = 10, 0.5, 0.2  # Define the number of generations, crossover probability, and mutation probability
algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None)  # Run the GA optimization

# Step 3: Train the Multi-Layer Perceptron (MLP) with Optimized Hyperparameters
best_individual = tools.selBest(population, 1)[0]  # Select the best individual from the population
best_n_neurons = int(best_individual[0])  # Ensure the number of neurons is an integer
best_alpha = abs(best_individual[1])  # Ensure alpha is non-negative

# Train the final MLP model using the best hyperparameters found by the GA
# The final model has stronger regularization (alpha) and a simpler architecture (fewer neurons)
final_model = MLPClassifier(hidden_layer_sizes=(best_n_neurons,), alpha=best_alpha, max_iter=2000, random_state=42)
final_model.fit(X_train, y_train)  # Train the MLP model on the training data

# Step 4: Evaluate the final model's performance
final_accuracy = final_model.score(X_test, y_test)  # Compute the accuracy of the final model on test data
print(f"Final model accuracy: {final_accuracy:.4f}")  # Print the final accuracy

# Compute the AUC-ROC
y_probs = final_model.predict_proba(X_test)[:, 1]  # Get probability estimates for the positive class
final_auc_roc = roc_auc_score(y_test, y_probs)  # Compute AUC-ROC
print(f"Final model AUC-ROC: {final_auc_roc:.4f}")  # Print the AUC-ROC

# Step 5: SHAP Interpretability

# Reduce the number of background samples using shap.sample() or shap.kmeans()
# For example, using shap.sample() to randomly select 100 background samples:
background_data = shap.sample(X_train, 100)  # Replace 100 with a suitable number for your use case

# Alternatively, using shap.kmeans() to create K representative background samples:
# background_data = shap.kmeans(X_train, 50)  # For instance, using 50 representative clusters

# Create the SHAP explainer with the reduced background set
explainer = shap.KernelExplainer(final_model.predict, background_data)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Plot the SHAP summary plot
shap.summary_plot(shap_values, X_test)
