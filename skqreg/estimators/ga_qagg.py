from typing import Optional

import numpy as np
from pygad import pygad
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler


class GAQuantizedAgg(BaseEstimator, ClassifierMixin):
    def __init__(self, n_levels=10, popsize=500):
        self.n_levels = n_levels
        self.popsize = popsize

        self.scaler = None  # type: Optional[MinMaxScaler]
        self.d = None
        self.classes_ = None
        self.cuts = None

    def fit(self, X, y):
        self.d = X.shape[1]
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.classes_ = np.unique(y)
        y_ = np.array(y == self.classes_[1], dtype=int)

        def objective(ga_instance, solution, solution_idx):
            # solution is a list of cuts
            cuts = [self._unpack(v) for v in solution]

            y_prob = self._predict_proba(X, cuts)
            # this is a variant of the expected entropy that multiplies with the absolute prediction error,
            # to enforce we actually try to predict the correct class end not only minimize entropy
            return 1 / (log_loss(y_, y_prob) + 1e-5)
            # return 1 / (expected_entropy_loss(y_prob) + 5 * mean_squared_error(y_, y_prob) + 1e-5)

        ga_instance = pygad.GA(num_generations=self.popsize,
                               num_parents_mating=5,
                               fitness_func=objective,
                               mutation_num_genes=1,
                               sol_per_pop=10,
                               num_genes=self.n_levels,
                               gene_space=[dict(low=-self.d, high=self.d) for _ in range(self.n_levels)]
                               )

        ga_instance.run()
        solution, fitness, _ = ga_instance.best_solution()
        self.cuts = [self._unpack(v) for v in solution]
        return self

    def _unpack(self, v):
        sgn = 1 if v >= 0 else -1
        v *= sgn
        return int(v), sgn, v % 1

    def _pack(self, f, sgn, thresh):
        return sgn * (f - 1 + thresh)

    @staticmethod
    def _predict_proba(X, cuts):
        results = []
        for f, sgn, cut in cuts:
            results.append(X[:, f] * sgn >= cut)
        results = np.array(results, dtype=int)
        return results.sum(axis=0) / len(cuts)

    def predict_proba(self, X):
        if self.cuts is None:
            raise NotFittedError()
        X = self.scaler.transform(X)
        y_prob = self._predict_proba(X, self.cuts)
        return np.vstack([1 - y_prob, y_prob]).T

    def predict(self, X):
        if self.cuts is None:
            raise NotFittedError()
        return self.classes_[np.array(self.predict_proba(X)[:, 1] >= .5, dtype=int)]


if __name__ == '__main__':
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml(data_id=42900, return_X_y=True, as_frame=False)

    print(cross_val_score(GAQuantizedAgg(), X, y, n_jobs=1, cv=2).mean())
