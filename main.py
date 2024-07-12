from knn import CustomKNN
from genetic_algorithm import GeneticAlgorithm
from data_processing import DataProcessor

def main():
    # Load and preprocess data
    data_processor = DataProcessor("battery_data.csv")
    X_train, y_train = data_processor.preprocess_data()

    # Initialize and optimize K-NN with genetic algorithm
    knn = CustomKNN()
    ga = GeneticAlgorithm(knn, X_train, y_train, population_size=50, generations=100)
    best_knn = ga.optimize()

    # Print the best K-NN configuration
    print("Best K-NN Configuration:", best_knn.get_params())

if __name__ == "__main__":
    main()
