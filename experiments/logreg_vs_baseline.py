from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import seaborn as sns
from skqreg.estimators import QuantizationAgg
from skqreg.estimators.ga_qagg import GAQuantizedAgg

if __name__ == '__main__':
    for dataset in [43979, 42900]:
        metric =        "roc_auc"
        print(dataset)
        X, y = fetch_openml(data_id=dataset, return_X_y=True, as_frame=False)
        for clf in (LogisticRegression(max_iter=500), QuantizationAgg(), GAQuantizedAgg()):
            results = cross_val_score(clf, X, y, n_jobs=-1, scoring=metric)
            print(results.mean(), sns.utils.ci(sns.algorithms.bootstrap(results)))
