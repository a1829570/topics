"""
This Python script implements a hybrid model using Genetic Algorithms (GA) and a Multi-Layer Perceptron (MLP) to predict cardiovascular disease (CVD) risk.
The preprocessing stage has been enhanced to dynamically handle numerical and categorical features across different datasets.
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
from deap import base, creator, tools, algorithms
import random
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Step 1: Load and Preprocess the Data

# Dynamically select numerical and categorical features based on the dataset's columns
def preprocess_data(dataset, target_col):
    # Drop target column and detect types dynamically
    X = dataset.drop(columns=[target_col])
    y = dataset[target_col]
    
    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Create a ColumnTransformer for dynamic preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
                ('scaler', StandardScaler())  # Apply StandardScaler after imputation
            ]), numerical_columns),  # Dynamic handling of numerical columns

            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values in categorical columns
                ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical columns
            ]), categorical_columns)  # Dynamic handling of categorical columns
        ]
    )

    # Apply the preprocessing pipeline
    X_transformed = preprocessor.fit_transform(X)
    
    return X_transformed, y, preprocessor

# Example usage with the original dataset (heart_disease_uci.csv)
original_data = pd.read_csv('heart_disease_uci.csv')
original_data['num'] = original_data['num'].apply(lambda x: 1 if x > 0 else 0)
X_original, y_original, original_preprocessor = preprocess_data(original_data, target_col='num')

# Example usage with another dataset (heart_failure_clinical_records_dataset.csv)
heart_failure_data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
X_heart_failure, y_heart_failure, heart_failure_preprocessor = preprocess_data(heart_failure_data, target_col='DEATH_EVENT')

# Define the MLP model function with Genetic Algorithm optimization
def ga_mlp_model(X_train, y_train, n_generations=10, population_size=20):
    # Define the Genetic Algorithm parameters and problem setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: float(random.uniform(0.01, 0.5)))  # Ensure learning rate range is positive
    toolbox.register("attr_int", lambda: int(random.randint(50, 200)))  # Number of neurons range as integer
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_float, toolbox.attr_int), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    # Define a mutation function that ensures learning rate stays positive
    def safe_mutation(individual):
        # Mutate each attribute while keeping learning rate positive
        tools.mutPolynomialBounded(individual, low=[0.01, 50], up=[0.5, 200], eta=0.1, indpb=0.2)
        # Safety check: Ensure learning rate remains above 0.01 and remove any complex parts
        individual[0] = max(0.01, float(abs(individual[0].real)))  # Take absolute value and ensure it's a float
        return individual,

    toolbox.register("mutate", safe_mutation)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_mlp_individual, X=X_train, y=y_train)

    # Genetic Algorithm evolution
    pop = toolbox.population(n=population_size)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations, verbose=False)
    
    # Extract the best individual and create the model
    best_ind = tools.selBest(pop, k=1)[0]
    best_learning_rate, best_neurons = float(best_ind[0]), int(best_ind[1])  # Ensure types are strictly float and int
    print(f"Best Individual: Learning Rate = {best_learning_rate}, Neurons = {best_neurons}")
    
    model = MLPClassifier(hidden_layer_sizes=(best_neurons,), learning_rate_init=best_learning_rate, max_iter=300)
    model.fit(X_train, y_train)
    return model


# Function to evaluate an MLP model individual
def evaluate_mlp_individual(individual, X, y):
    learning_rate, neurons = individual
    model = MLPClassifier(hidden_layer_sizes=(int(neurons),), learning_rate_init=learning_rate, max_iter=300)
    model.fit(X, y)
    return (roc_auc_score(y, model.predict_proba(X)[:, 1]),)

# Train the model on the original dataset and evaluate it on the heart failure dataset
X_train, X_test, y_train, y_test = train_test_split(X_original, y_original, test_size=0.2, random_state=42)
mlp_model = ga_mlp_model(X_train, y_train)
y_pred = mlp_model.predict(X_test)

# Evaluate on the test set of the original dataset
print(f"Accuracy on Original Dataset: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report on Original Dataset:")
print(classification_report(y_test, y_pred))

# Test the model on the heart failure dataset
X_heart_failure_test_transformed = original_preprocessor.transform(heart_failure_data.drop(columns=['DEATH_EVENT']))
y_heart_failure_test = heart_failure_data['DEATH_EVENT']
y_heart_failure_pred = mlp_model.predict(X_heart_failure_test_transformed)

# Evaluate on the heart failure dataset
print(f"Accuracy on Heart Failure Dataset: {accuracy_score(y_heart_failure_test, y_heart_failure_pred) * 100:.2f}%")
print("Classification Report on Heart Failure Dataset:")
print(classification_report(y_heart_failure_test, y_heart_failure_pred))
