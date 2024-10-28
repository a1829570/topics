
"""This Python script implements a hybrid model using Genetic Algorithms (GA) and a Multi-Layer Perceptron (MLP) to predict cardiovascular disease (CVD) risk.
"""

# Import necessary libraries
import pandas as pd  # For data handling and manipulation
import numpy as np  # For numerical computations
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # To standardize features and encode categorical variables
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer  # To handle missing values
from sklearn.neural_network import MLPClassifier  # To implement the Multi-Layer Perceptron model
from deap import base, creator, tools, algorithms  # To implement the Genetic Algorithm
import random  # For random number generation, used by the GA
import shap  # Import SHAP for interpretability analysis
from sklearn.metrics import roc_auc_score  # To compute AUC-ROC

# Step 1: Load and Preprocess the Data
data = pd.read_csv('heart_disease_uci.csv')  # Load the dataset from the file

# Convert 'num' to binary classification (0: no disease, 1: disease)
data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0) 

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
def evaluate(individual):
    n_neurons = int(individual[0])  # Ensure n_neurons is an integer
    alpha = abs(individual[1])  # Ensure alpha is non-negative

    # Initialize the MLPClassifier with the current hyperparameters
    mlp = MLPClassifier(hidden_layer_sizes=(n_neurons,), alpha=alpha, max_iter=2000, random_state=42)  # Set random_state for reproducibility
    mlp.fit(X_train, y_train)  # Train the MLP model on the training data

    # Evaluate the model's performance using accuracy on the test set
    accuracy = mlp.score(X_test, y_test)  # Compute accuracy
    return accuracy,  # Return a tuple as required by the DEAP library

# Set up the DEAP library for the Genetic Algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Create a fitness function to maximize accuracy
creator.create("Individual", list, fitness=creator.FitnessMax)  # Define an individual as a list with fitness attribute

# Register genetic algorithm operators for individuals
toolbox = base.Toolbox()  # Initialize the toolbox for GA operations
toolbox.register("attr_int", random.randint, 50, 200)  # Register an integer attribute for the number of neurons (50 to 200)
toolbox.register("attr_float", random.uniform, 0.0001, 1.0)  # Register a float attribute for the alpha parameter (0.0001 to 1.0)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_int, toolbox.attr_float), n=1)  # Register how to create individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Register how to create a population of individuals

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
final_model = MLPClassifier(hidden_layer_sizes=(best_n_neurons,), alpha=best_alpha, max_iter=2000, random_state=42)  # Set random_state
final_model.fit(X_train, y_train)  # Train the MLP model on the training data

# Evaluate the final model's performance
final_accuracy = final_model.score(X_test, y_test)  # Compute the accuracy of the final model
print(f"Final model accuracy: {final_accuracy:.4f}")  # Print the final accuracy

# Compute the AUC-ROC
y_probs = final_model.predict_proba(X_test)[:, 1]  # Get probability estimates for the positive class
final_auc_roc = roc_auc_score(y_test, y_probs)  # Compute AUC-ROC
print(f"Final model AUC-ROC: {final_auc_roc:.4f}")  # Print the AUC-ROC

# Create a SHAP explainer for the MLP model using KernelExplainer
explainer = shap.KernelExplainer(final_model.predict, X_train)  # Initialize SHAP explainer with the model's prediction function
shap_values = explainer.shap_values(X_test)  # Compute SHAP values for the test set

# Plot the SHAP summary plot to interpret model predictions
shap.summary_plot(shap_values, X_test)  # Plot summary of SHAP values
