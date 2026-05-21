from sklearn.model_selection._split import _BaseKFold
import pandas as pd
import numpy as np


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
