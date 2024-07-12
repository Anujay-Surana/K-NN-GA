import numpy as np

class GeneticAlgorithm:
    def __init__(self, model, X, y, population_size=50, generations=100):
        self.model = model
        self.X = X
        self.y = y
        self.population_size = population_size
        self.generations = generations

    def optimize(self):
        population = self._initialize_population()
        for generation in range(self.generations):
            fitness_scores = self._evaluate_population(population)
            population = self._select_population(population, fitness_scores)
            population = self._crossover_population(population)
            population = self._mutate_population(population)
        best_individual = self._select_best_individual(population)
        self.model.set_params(**best_individual)
        return self.model

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {
                'n_neighbors': np.random.randint(1, 20),
                'weights': np.random.choice(['uniform', 'distance']),
                'p': np.random.randint(1, 3)
            }
            population.append(individual)
        return population

    def _evaluate_population(self, population):
        fitness_scores = []
        for individual in population:
            self.model.set_params(**individual)
            self.model.fit(self.X, self.y)
            accuracy = self._evaluate_fitness(self.X, self.y)
            fitness_scores.append(accuracy)
        return fitness_scores

    def _evaluate_fitness(self, X, y):
        predictions = self.model.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def _select_population(self, population, fitness_scores):
        selected_population = np.random.choice(population, size=self.population_size, p=fitness_scores/np.sum(fitness_scores))
        return selected_population.tolist()

    def _crossover_population(self, population):
        new_population = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[(i+1) % len(population)]
            crossover_point = np.random.randint(1, len(parent1))
            child1 = {**parent1}
            child2 = {**parent2}
            for j, key in enumerate(parent1.keys()):
                if j >= crossover_point:
                    child1[key] = parent2[key]
                    child2[key] = parent1[key]
            new_population.append(child1)
            new_population.append(child2)
        return new_population

    def _mutate_population(self, population):
        for individual in population:
            if np.random.rand() < 0.1:
                key = np.random.choice(list(individual.keys()))
                if key == 'n_neighbors':
                    individual[key] = np.random.randint(1, 20)
                elif key == 'weights':
                    individual[key] = np.random.choice(['uniform', 'distance'])
                elif key == 'p':
                    individual[key] = np.random.randint(1, 3)
        return population

    def _select_best_individual(self, population):
        best_individual = None
        best_fitness = -np.inf
        for individual in population:
            self.model.set_params(**individual)
            self.model.fit(self.X, self.y)
            fitness = self._evaluate_fitness(self.X, self.y)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual
        return best_individual
