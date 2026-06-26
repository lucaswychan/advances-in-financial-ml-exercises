import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _get_ivp(cov):
    """Return inverse-variance portfolio weights for a covariance matrix."""

    variances = np.diag(np.asarray(cov, dtype=float))
    if np.any(variances <= 0) or np.any(~np.isfinite(variances)):
        raise ValueError("Covariance matrix diagonal must contain positive finite values.")

    inverse_variances = 1.0 / variances
    return inverse_variances / inverse_variances.sum()


def _get_cluster_var(cov, c_items):
    """Return the inverse-variance-weighted variance of a cluster."""

    if len(c_items) == 0:
        raise ValueError("Cluster must contain at least one asset.")

    sub_cov = cov.iloc[c_items, c_items].to_numpy(dtype=float)
    weights = _get_ivp(sub_cov).reshape(-1, 1)
    return (weights.T @ sub_cov @ weights).item()


class HierarchicalRiskParity:
    def __init__(self):
        self.returns = None
        self.cov = None
        self.corr = None
        self.clusters = None
        self.ordered_indices = None
        self.weights = None

    def tree_clustering(self, corr) -> np.ndarray:
        """Build the hierarchical clustering tree from a correlation matrix."""

        corr_values = np.asarray(corr, dtype=float)
        if corr_values.ndim != 2 or corr_values.shape[0] != corr_values.shape[1]:
            raise ValueError("Correlation matrix must be square.")
        if corr_values.shape[0] < 2:
            raise ValueError("At least two assets are required for hierarchical clustering.")
        if np.any(~np.isfinite(corr_values)):
            raise ValueError("Correlation matrix must contain only finite values.")

        distance = np.sqrt(np.clip((1.0 - corr_values) / 2.0, 0.0, 1.0))
        np.fill_diagonal(distance, 0.0)
        return linkage(squareform(distance, checks=False), method="single")

    def get_quasi_diag(self, clusters) -> list:
        """Return the leaf order implied by the hierarchical clustering tree."""

        clusters = np.asarray(clusters, dtype=int)
        if clusters.ndim != 2 or clusters.shape[1] != 4:
            raise ValueError("Linkage matrix must have shape (n - 1, 4).")

        num_items = int(clusters[-1, 3])

        def expand(node):
            node = int(node)
            if node < num_items:
                return [node]
            left, right = clusters[node - num_items, :2]
            return expand(left) + expand(right)

        return expand(clusters[-1, 0]) + expand(clusters[-1, 1])

    @staticmethod
    def _split_cluster(cluster):
        midpoint = len(cluster) // 2
        return cluster[:midpoint], cluster[midpoint:]

    def recursive_bisection(self, assets, cov, ordered_indices) -> pd.DataFrame:
        """Allocate weights recursively across the ordered HRP tree leaves."""

        ordered_indices = list(ordered_indices)
        weights = pd.Series(1.0, index=ordered_indices, dtype=float)
        clusters = [ordered_indices]

        while clusters:
            next_clusters = []
            for cluster in clusters:
                left_cluster, right_cluster = self._split_cluster(cluster)
                next_clusters.extend([left_cluster, right_cluster])
            clusters = [cluster for cluster in next_clusters if cluster]

            for left_cluster, right_cluster in zip(clusters[0::2], clusters[1::2]):
                left_variance = _get_cluster_var(cov, left_cluster)
                right_variance = _get_cluster_var(cov, right_cluster)
                total_variance = left_variance + right_variance
                if total_variance <= 0:
                    raise ValueError("Cluster variances must sum to a positive value.")

                left_allocation = 1.0 - left_variance / total_variance
                weights.loc[left_cluster] *= left_allocation
                weights.loc[right_cluster] *= 1.0 - left_allocation

            clusters = [cluster for cluster in clusters if len(cluster) > 1]

        asset_order = assets.columns.take(weights.index)
        return pd.DataFrame([weights.to_numpy(dtype=float)], columns=asset_order)

    def allocate(self, assets, resample_by="W") -> None:
        if not isinstance(assets, pd.DataFrame):
            raise ValueError("Assets matrix must be a dataframe.")
        if not isinstance(assets.index, pd.DatetimeIndex):
            raise ValueError("Assets dataframe must be indexed by date.")

        resampled_assets = assets.resample(resample_by).last()
        self.returns = resampled_assets.pct_change().dropna(how="all")
        if self.returns.empty:
            raise ValueError("At least one return observation is required.")

        self.cov = self.returns.cov()
        self.corr = self.returns.corr()
        self.clusters = self.tree_clustering(self.corr)
        self.ordered_indices = self.get_quasi_diag(self.clusters)
        self.weights = self.recursive_bisection(resampled_assets, self.cov, self.ordered_indices)
