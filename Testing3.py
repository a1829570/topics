
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
from sklearn.metrics import accuracy_score, classification_report

# Function to identify numerical and categorical columns dynamically
def get_column_types(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return numerical_columns, categorical_columns

# Function to create a dynamic preprocessing pipeline based on the dataset's columns
def create_preprocessing_pipeline(df):
    numerical_columns, categorical_columns = get_column_types(df)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ]
    )
    return preprocessor

# Load both datasets
uci_data = pd.read_csv('heart_disease_uci.csv')
uci_data['num'] = uci_data['num'].apply(lambda x: 1 if x > 0 else 0)
X_uci = uci_data.drop('num', axis=1)
y_uci = uci_data['num']

hf_data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
X_hf = hf_data.drop('DEATH_EVENT', axis=1)
y_hf = hf_data['DEATH_EVENT']

# Create a preprocessing pipeline for the UCI dataset
preprocessor_uci = create_preprocessing_pipeline(X_uci)
X_uci_train, X_uci_test, y_uci_train, y_uci_test = train_test_split(X_uci, y_uci, test_size=0.2, random_state=42)
X_uci_train = preprocessor_uci.fit_transform(X_uci_train)
X_uci_test = preprocessor_uci.transform(X_uci_test)

# Create a preprocessing pipeline for the Heart Failure dataset
preprocessor_hf = create_preprocessing_pipeline(X_hf)
X_hf_train, X_hf_test, y_hf_train, y_hf_test = train_test_split(X_hf, y_hf, test_size=0.2, random_state=42)
X_hf_train = preprocessor_hf.fit_transform(X_hf_train)
X_hf_test = preprocessor_hf.transform(X_hf_test)

# Define the GA-MLP Hybrid Model
def hybrid_model_GA_MLP(X_train, y_train, X_test, y_test, dataset_name=""):
    # Create MLP Classifier with DEAP for GA optimization
    def evaluate(individual):
        hidden_layer_size = int(individual[0])
        model = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), max_iter=300, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return accuracy_score(y_test, predictions),

    # DEAP GA setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 50, 200)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=50, up=200, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # Run GA
    population = toolbox.population(n=20)
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, verbose=False)

    # Get the best individual
    best_individual = tools.selBest(population, k=1)[0]
    print(f"Best Individual for {dataset_name}: {best_individual}")

    # Final evaluation
    best_hidden_layer_size = int(best_individual[0])
    final_model = MLPClassifier(hidden_layer_sizes=(best_hidden_layer_size,), max_iter=300, random_state=42)
    final_model.fit(X_train, y_train)
    final_predictions = final_model.predict(X_test)
    print(f"Final Model Accuracy for {dataset_name}:", accuracy_score(y_test, final_predictions))
    print(f"Classification Report for {dataset_name}:\n", classification_report(y_test, final_predictions))


# Execute GA-MLP hybrid model on both datasets separately
print("Evaluating on UCI Heart Disease Dataset:")
hybrid_model_GA_MLP(X_uci_train, y_uci_train, X_uci_test, y_uci_test, dataset_name="UCI Heart Disease")

print("Evaluating on Heart Failure Dataset:")
hybrid_model_GA_MLP(X_hf_train, y_hf_train, X_hf_test, y_hf_test, dataset_name="Heart Failure")
