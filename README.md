# K-NN-GA
A combination of K-NN's optimized by Genetic Algorithms for Euclidean Alternative Training.

# Proprietary K-NN Algorithm with Genetic Algorithms for Grouping Battery Batches

## Project Overview

This project implements a custom K-Nearest Neighbors (K-NN) algorithm optimized using genetic algorithms to group battery batches based on performance metrics. The goal is to enhance the accuracy and efficiency of the K-NN algorithm by optimizing hyperparameters and feature weights.

## File Structure
project/
│
├── main.py
├── knn.py
├── genetic_algorithm.py
└── data_processing.py



## Description of Files

- **main.py**: Orchestrates the workflow by loading data, initializing the models, and running the genetic algorithm for optimization.
- **knn.py**: Defines the custom K-NN algorithm with methods for fitting, predicting, and parameter setting.
- **genetic_algorithm.py**: Implements the genetic algorithm for optimizing the K-NN model, including population initialization, evaluation, selection, crossover, and mutation.
- **data_processing.py**: Handles data loading, preprocessing, and splitting into training and testing sets.

## Requirements

- Python 3.x
- NumPy
- Pandas
- SciPy
- Scikit-learn

You can install the required packages using pip:

```bash
pip install numpy pandas scipy scikit-learn


Usage
1. Data Preparation
Ensure your battery performance data is in a CSV file with appropriate columns. The target variable (e.g., battery batch labels) should be in a column named target.

2. Preprocess Data
The DataProcessor class in data_processing.py handles data loading and preprocessing. It scales the features and splits the data into training and testing sets.

from data_processing import DataProcessor
data_processor = DataProcessor("battery_data.csv")
X_train, y_train, X_test, y_test = data_processor.preprocess_data()

3. Custom K-NN Algorithm
The CustomKNN class in knn.py defines the custom K-NN algorithm. It includes methods for fitting, predicting, and setting parameters.

from knn import CustomKNN
knn = CustomKNN()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

4. Genetic Algorithm for Optimization
The GeneticAlgorithm class in genetic_algorithm.py implements the genetic algorithm to optimize the K-NN model. It initializes the population, evaluates fitness, selects individuals, performs crossover, and mutation.
from genetic_algorithm import GeneticAlgorithm

ga = GeneticAlgorithm(knn, X_train, y_train, population_size=50, generations=100)
best_knn = ga.optimize()
print("Best K-NN Configuration:", best_knn.get_params())


5. Running the Project
You can run the entire workflow by executing main.py. This script loads the data, initializes the models, runs the genetic algorithm, and prints the best K-NN configuration.
python main.py


Contact
For any questions or issues, please contact the project maintainer at anujaysurana1@gmail.com



