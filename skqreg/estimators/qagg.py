from bisect import bisect_left

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class QuantizationAgg(BaseEstimator, ClassifierMixin):
    def __init__(self, n_levels=10, random_state=None):
        self.n_levels = n_levels
        self.random_state = random_state
        self.classes_ = None
        self.cuts = None

    def fit(self, X, y):
        clf = LogisticRegression(max_iter=500, random_state=self.random_state).fit(X, y)

        y_ = np.array(y == clf.classes_[1], dtype=int)
        c = clf.coef_[0]
        c = c / np.abs(c).sum() * self.n_levels

        n_cuts = np.round(c).astype(int)
        if abs(n_cuts).sum() != self.n_levels:
            for i in abs(c) % 1:
                if np.round(np.abs(c + i)).sum() == self.n_levels:
                    n_cuts = np.round(c + i).astype(int)
                    break

        self.cuts = []
        for i, c in enumerate(n_cuts):
            if c == 0:
                continue
            iso = IsotonicRegression().fit(X[:, i], y_)
            for split in np.linspace(0, 1, num=abs(c) + 2)[1:-1]:
                # TODO check decision boundaries in more detail
                thresh = iso.X_thresholds_[min(bisect_left(iso.y_thresholds_, split), len(iso.X_thresholds_) - 1)]
                self.cuts.append([i, 1 if c >= 0 else -1, thresh])
        self.classes_ = clf.classes_
        return self

    def predict(self, X):
        if self.cuts is None:
            raise NotFittedError()
        return self.classes_[np.array(self.predict_proba(X)[:, 1] >= .5, dtype=int)]

    def predict_proba(self, X):
        if self.cuts is None:
            raise NotFittedError()
        results = []
        for f, sgn, cut in self.cuts:
            results.append(X[:, f] * sgn >= cut)
        results = np.array(results, dtype=int)
        proba_pos = results.sum(axis=0) / self.n_levels
        return np.vstack([1 - proba_pos, proba_pos]).T


if __name__ == '__main__':
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml(data_id=43979, return_X_y=True, as_frame=False)

    print(cross_val_score(QuantizationAgg(random_state=42), X, y, n_jobs=1, cv=2).mean())
