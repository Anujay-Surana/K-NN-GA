import numpy as np

class CustomKNN:
    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = self._compute_distances(X)
        return self._vote(distances)

    def _compute_distances(self, X):
        distances = np.zeros((X.shape[0], self.X_train.shape[0]))
        for i, x in enumerate(X):
            for j, x_train in enumerate(self.X_train):
                distances[i, j] = np.linalg.norm(x - x_train, ord=self.p)
        return distances

    def _vote(self, distances):
        y_pred = np.zeros(distances.shape[0])
        for i, d in enumerate(distances):
            neighbors = np.argsort(d)[:self.n_neighbors]
            neighbor_votes = self.y_train[neighbors]
            y_pred[i] = np.argmax(np.bincount(neighbor_votes))
        return y_pred

    def get_params(self):
        return {'n_neighbors': self.n_neighbors, 'weights': self.weights, 'p': self.p}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
