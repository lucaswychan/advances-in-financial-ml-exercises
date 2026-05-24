from sklearn.model_selection._split import _BaseKFold
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss


class PurgedKFold(_BaseKFold):
    """
    Extend KFold to work with labels that span intervals
    The train is is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training examples in between
    """

    def __init__(self, n_splits=3, t1=None, pct_embargo=0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("Label through Dates must be a pandas series")
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        if X.shape[0] != self.t1.shape[0]:
            raise ValueError("X and ThruDateValues must have the same index length")
        if not X.index.equals(self.t1.index):
            raise ValueError("X and ThruDateValues must have the same index")
        indices = np.arange(X.shape[0])
        embargo = int(X.shape[0] * self.pct_embargo)
        test_starts = [
            (i[0], i[-1] + 1)
            for i in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]
        for i, j in test_starts:
            t0 = self.t1.index[i]
            test_indices = indices[i:j]
            # test_indices are positional (from sklearn splitters), so use iloc.
            max_t1 = self.t1.iloc[test_indices].max()
            maxT1Idx = self.t1.index.searchsorted(max_t1)
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            train_indices = np.concatenate(
                (train_indices, indices[maxT1Idx + embargo :])
            )
            yield train_indices, test_indices


def _score_model(model, X, y, sample_weight=None, scoring_metric="accuracy"):
    supported_metrics = ["accuracy", "neg_log_loss"]
    if scoring_metric not in supported_metrics:
        raise ValueError(f"Invalid scoring metric. Supported metrics are {supported_metrics}.")

    if scoring_metric == "accuracy":
        predictions = model.predict(X)
        return accuracy_score(y, predictions, sample_weight=sample_weight)

    probabilities = model.predict_proba(X)
    return -log_loss(
        y,
        probabilities,
        sample_weight=sample_weight,
        labels=model.classes_,
    )


def cross_val_scores(model, X, y, cv_generator, sample_weight=None, scoring_metric="accuracy"):
    supported_metrics = ["accuracy", "neg_log_loss"]
    if scoring_metric not in supported_metrics:
        raise ValueError(f"Invalid scoring metric. Supported metrics are {supported_metrics}.")

    scores = []
    for train_indices, test_indices in cv_generator.split(X, y):
        X_train, X_test = X.iloc[train_indices, :], X.iloc[test_indices, :]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        w_train = None if sample_weight is None else sample_weight.iloc[train_indices].values
        w_test = None if sample_weight is None else sample_weight.iloc[test_indices].values

        fit = model.fit(X_train, y_train, sample_weight=w_train)
        scores.append(_score_model(fit, X_test, y_test, w_test, scoring_metric))

    return pd.Series(scores)