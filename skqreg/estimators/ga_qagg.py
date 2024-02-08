from typing import Optional

import numpy as np
from pygad import pygad
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler


def _unpack(v):
    sgn = 1 if v >= 0 else -1
    return int(v), sgn * ((v * sgn) % 1)


class GAQuantizedAgg(BaseEstimator, ClassifierMixin):
    def __init__(self, n_levels=10, popsize=500, random_state=None):
        self.random_state = random_state
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

        def objective(instance, solution_, solution_idx):
            # solution is a list of cuts
            cuts = [_unpack(v) for v in solution_]

            y_prob = self._predict_proba(X, cuts)
            # cross-entropy loss
            return 1 / (log_loss(y_, y_prob) + 1e-5)

        ga_instance = pygad.GA(num_generations=self.popsize,
                               num_parents_mating=5,
                               fitness_func=objective,
                               mutation_num_genes=1,
                               sol_per_pop=10,
                               num_genes=self.n_levels,
                               gene_space=[dict(low=-self.d, high=self.d) for _ in range(self.n_levels)],
                               random_seed=self.random_state
                               )

        ga_instance.run()
        solution, fitness, _ = ga_instance.best_solution()

        self.cuts = []
        for f, t in [_unpack(v) for v in solution]:
            a = np.zeros((1, self.d))
            a[0, f] = t
            t = self.scaler.inverse_transform(a)[0,f]
            self.cuts.append((f, t))
        return self

    @staticmethod
    def _predict_proba(X, cuts):
        results = []
        for f, cut in cuts:
            results.append(X[:, f] >= cut)
        results = np.array(results, dtype=int)
        return results.sum(axis=0) / len(cuts)

    def predict_proba(self, X):
        if self.cuts is None:
            raise NotFittedError()
        y_prob = self._predict_proba(X, self.cuts)
        return np.vstack([1 - y_prob, y_prob]).T

    def predict(self, X):
        if self.cuts is None:
            raise NotFittedError()
        return self.classes_[np.array(self.predict_proba(X)[:, 1] >= .5, dtype=int)]


if __name__ == '__main__':
    from sklearn.datasets import fetch_openml

    X_y = fetch_openml(data_id=42900, return_X_y=True, as_frame=False)
    clf = GAQuantizedAgg(random_state=42).fit(*X_y)
    score = clf.score(*X_y)
    print(score)
    print(clf.cuts)
