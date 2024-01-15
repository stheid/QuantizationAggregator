# Quantistic Regression

This package provides a interpretable binary classifier, inspired by Scoring Systems.
Unlike Scoring Systems, however, this classifier will not predict scores with a deterministic decition threshold, but it will predict discrete probability levels, making it an interpretable probabilistic classifier.

This implementation adheres to the [sklearn-api](https://scikit-learn.org/stable/glossary.html#glossary-estimator-types).
To this end, the discrete probability levels are turned into real valued probability levels to be technically compatible with other probabilistic classifiers.