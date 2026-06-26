# from https://github.com/hudson-and-thames/mlfinlab
# license: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

'''
This module implements the famous Critical Line Algorithm for mean-variance portfolio
optimisation. It is reproduced with modification from the following paper:
`D.H. Bailey and M.L. Prado “An Open-Source Implementation of the Critical- Line Algorithm for
Portfolio Optimization”,Algorithms, 6 (2013), 169-196. <http://dx.doi.org/10.3390/a6010169>`_
'''

import numbers
from math import log, ceil
import numpy as np
import pandas as pd


SUPPORTED_RETURN_METHODS = {"mean", "exponential"}
SUPPORTED_SOLUTIONS = {"cla_turning_points", "efficient_frontier", "min_volatility", "max_sharpe"}


class CriticalLineAlgorithm:
    # pylint: disable=too-many-instance-attributes
    '''
    CLA is a famous portfolio optimisation algorithm used for calculating the optimal allocation weights for a given
    portfolio. It solves the optimisation problem with constraints on each weight - lower and upper bounds on the weight
    value. This class can compute multiple types of solutions - the normal cla solution, minimum variance solution,
    maximum sharpe solution and finally the solution to the efficient frontier.
    '''

    def __init__(self, weight_bounds=(0, 1), calculate_returns="mean"):
        '''
        Initialise the storage arrays and some preprocessing.

        :param weight_bounds: (tuple) a tuple specifying the lower and upper bound ranges for the portfolio weights
        :param calculate_returns: (str) the method to use for calculation of expected returns.
                                        Currently supports "mean" and "exponential"
        '''

        self.weight_bounds = weight_bounds
        self.calculate_returns = calculate_returns
        self.weights = list()
        self.lambdas = list()
        self.gammas = list()
        self.free_weights = list()
        self.expected_returns = None
        self.cov_matrix = None
        self.lower_bounds = None
        self.upper_bounds = None
        self.max_sharpe = None
        self.min_var = None
        self.efficient_frontier_means = None
        self.efficient_frontier_sigma = None

    @staticmethod
    def _infnone(number):
        '''
        Converts a Nonetype object to inf

        :param number: (int/float/None) a number
        :return: (float) -inf or number
        '''
        return float("-inf") if number is None else number

    @staticmethod
    def _as_scalar(value):
        '''
        Convert a NumPy scalar or single-value array to a Python scalar.
        '''

        return np.asarray(value).item()

    @property
    def _num_assets(self):
        return self.expected_returns.shape[0]

    @staticmethod
    def _to_column(values):
        return np.asarray(values, dtype=float).reshape(-1, 1)

    @staticmethod
    def _weights_to_frame(weights, assets):
        weights = [np.asarray(weight, dtype=float).reshape(-1) for weight in weights]
        return pd.DataFrame(weights, columns=assets)

    def _init_algo(self):
        '''
        Initial setting up of the algorithm. Calculates the first free weight of the first turning point.

        :return: (list, list) asset index and the corresponding free weight value
        '''

        ranked_assets = np.argsort(self.expected_returns.reshape(-1), kind="mergesort")
        weights = np.copy(self.lower_bounds)

        for asset in ranked_assets[::-1]:
            weights[asset] = self.upper_bounds[asset]
            if np.sum(weights) >= 1:
                weights[asset] += 1 - np.sum(weights)
                return [asset], weights

        raise ValueError("Weight bounds do not allow a fully invested portfolio.")

    @staticmethod
    def _compute_bi(c_final, asset_bounds_i):
        '''
        Calculates which bound value to assign to a bounded asset - lower bound or upper bound.

        :param c_final: (float) a value calculated using the covariance matrices of free weights.
                          Refer to https://pdfs.semanticscholar.org/4fb1/2c1129ba5389bafe47b03e595d098d0252b9.pdf for
                          more information.
        :param asset_bounds_i: (list) a list containing the lower and upper bound values for the ith weight
        :return: bounded weight value
        '''

        if c_final > 0:
            return CriticalLineAlgorithm._as_scalar(asset_bounds_i[1])
        return CriticalLineAlgorithm._as_scalar(asset_bounds_i[0])

    def _compute_w(self, covar_f_inv, covar_fb, mean_f, w_b):
        '''
        Compute the turning point associated with the current set of free weights F

        :param covar_f_inv: (np.array) inverse of covariance matrix of free assets
        :param covar_fb: (np.array) covariance matrix between free assets and bounded assets
        :param mean_f: (np.array) expected returns of free assets
        :param w_b: (np.array) bounded asset weight values

        :return: (array, float) list of turning point weights and gamma value from the langrange equation
        '''

        # Compute gamma
        ones_f = np.ones(mean_f.shape)
        g_1 = self._as_scalar(ones_f.T @ covar_f_inv @ mean_f)
        g_2 = self._as_scalar(ones_f.T @ covar_f_inv @ ones_f)
        if w_b is None:
            g_final, w_1 = -self.lambdas[-1] * g_1 / g_2 + 1 / g_2, 0
        else:
            ones_b = np.ones(w_b.shape)
            g_3 = self._as_scalar(ones_b.T @ w_b)
            g_4 = covar_f_inv @ covar_fb
            w_1 = g_4 @ w_b
            g_4 = self._as_scalar(ones_f.T @ w_1)
            g_final = -self.lambdas[-1] * g_1 / g_2 + (1 - g_3 + g_4) / g_2

        # Compute weights
        w_2 = covar_f_inv @ ones_f
        w_3 = covar_f_inv @ mean_f
        free_asset_weights = -w_1 + g_final * w_2 + self.lambdas[-1] * w_3
        return free_asset_weights, g_final

    def _compute_lambda(self, covar_f_inv, covar_fb, mean_f, w_b, asset_index, b_i):
        '''
        Calculate the lambda value in the langrange optimsation equation

        :param covar_f_inv: (np.array) inverse of covariance matrix of free assets
        :param covar_fb: (np.array) covariance matrix between free assets and bounded assets
        :param mean_f: (np.array) expected returns of free assets
        :param w_b: (np.array) bounded asset weight values
        :param asset_index: (int) index of the asset in the portfolio
        :param b_i: (list) list of upper and lower bounded weight values
        :return: (float) lambda value
        '''

        # Compute C
        ones_f = np.ones(mean_f.shape)
        c_1 = self._as_scalar(ones_f.T @ covar_f_inv @ ones_f)
        c_2 = covar_f_inv @ mean_f
        c_3 = self._as_scalar(ones_f.T @ covar_f_inv @ mean_f)
        c_4 = covar_f_inv @ ones_f
        c_final = -1*c_1 * self._as_scalar(c_2[asset_index]) + c_3 * self._as_scalar(c_4[asset_index])
        if c_final == 0:
            return None, None

        # Compute bi
        if isinstance(b_i, list):
            b_i = self._compute_bi(c_final, b_i)
        else:
            b_i = self._as_scalar(b_i)

        # Compute Lambda
        if w_b is None:

            # All free assets
            return (self._as_scalar(c_4[asset_index]) - c_1 * b_i) / c_final, b_i

        ones_b = np.ones(w_b.shape)
        l_1 = self._as_scalar(ones_b.T @ w_b)
        l_2 = covar_f_inv @ covar_fb
        l_3 = l_2 @ w_b
        l_2 = self._as_scalar(ones_f.T @ l_3)
        lambda_value = (
            (1 - l_1 + l_2) * self._as_scalar(c_4[asset_index])
            - c_1 * (b_i + self._as_scalar(l_3[asset_index]))
        ) / c_final
        return lambda_value, b_i

    def _get_matrices(self, free_weights):
        '''
        Calculate the required matrices between free and bounded assets

        :param free_weights: (list) list of free assets/weights
        :return: (tuple of np.array matrices) the corresponding matrices
        '''

        covar_f = self._reduce_matrix(self.cov_matrix, free_weights, free_weights)
        mean_f = self._reduce_matrix(self.expected_returns, free_weights, [0])
        bounded_weights = self._get_bounded_weights(free_weights)
        covar_fb = self._reduce_matrix(self.cov_matrix, free_weights, bounded_weights)
        w_b = self._reduce_matrix(self.weights[-1], bounded_weights, [0])
        return covar_f, covar_fb, mean_f, w_b

    def _get_bounded_weights(self, free_weights):
        '''
        Compute the list of bounded assets

        :param free_weights: (np.array) list of free weights/assets
        :return: (np.array) list of bounded assets/weights
        '''

        return self._diff_lists(list(range(self._num_assets)), free_weights)

    @staticmethod
    def _diff_lists(list_1, list_2):
        '''
        Calculate the set difference between two lists

        :param list_1: (list) a list of asset indices
        :param list_2: (list) another list of asset indices
        :return: (list) set difference between the two input lists
        '''

        excluded = set(list_2)
        return [item for item in list_1 if item not in excluded]

    @staticmethod
    def _reduce_matrix(matrix, row_indices, col_indices):
        '''
        Reduce a matrix to the provided set of rows and columns

        :param matrix: (np.array) a matrix whose subset of rows and columns we need
        :param row_indices: (list) list of row indices for the matrix
        :param col_indices: (list) list of column indices for the matrix
        :return: (np.array) subset of input matrix
        '''

        return matrix[np.ix_(row_indices, col_indices)]

    def _purge_num_err(self, tol):
        '''
        Purge violations of inequality constraints (associated with ill-conditioned cov matrix)

        :param tol: (float) tolerance level for purging
        '''

        filtered = []
        for weight, lambda_value, gamma, free_weight in zip(
                self.weights, self.lambdas, self.gammas, self.free_weights
        ):
            fully_invested = abs(np.sum(weight) - 1) <= tol
            respects_bounds = np.all(weight >= self.lower_bounds - tol) and np.all(weight <= self.upper_bounds + tol)
            if fully_invested and respects_bounds:
                filtered.append((weight, lambda_value, gamma, free_weight))

        if not filtered:
            raise ValueError("No numerically valid CLA turning points remain after purging.")

        self.weights, self.lambdas, self.gammas, self.free_weights = map(list, zip(*filtered))

    def _purge_excess(self):
        '''
        Remove violations of the convex hull
        '''

        if len(self.weights) <= 2:
            return

        means = [self._as_scalar(weight.T @ self.expected_returns) for weight in self.weights]
        keep = [False] * len(self.weights)
        keep[0] = True

        highest_later_mean = float("-inf")
        for index in range(len(self.weights) - 1, 0, -1):
            if means[index] >= highest_later_mean:
                keep[index] = True
                highest_later_mean = means[index]

        self.weights = [weight for weight, should_keep in zip(self.weights, keep) if should_keep]
        self.lambdas = [value for value, should_keep in zip(self.lambdas, keep) if should_keep]
        self.gammas = [value for value, should_keep in zip(self.gammas, keep) if should_keep]
        self.free_weights = [value for value, should_keep in zip(self.free_weights, keep) if should_keep]

    @staticmethod
    def _golden_section(obj, left, right, **kwargs):
        '''
        Golden section method. Maximum if kargs['minimum']==False is passed

        :param obj: (function) The objective function on which the extreme will be found.
        :param left: (float) The leftmost extreme of search
        :param right: (float) The rightmost extreme of search
        '''

        tol = 1.0e-9
        sign = 1 if kwargs.get("minimum", True) else -1
        args = kwargs.get("args", ())
        if left == right:
            return left, obj(left, *args)
        num_iterations = int(ceil(-2.078087 * log(tol / abs(right - left))))
        gs_ratio = 0.618033989
        complementary_gs_ratio = 1.0 - gs_ratio

        # Initialize
        x_1 = gs_ratio * left + complementary_gs_ratio * right
        x_2 = complementary_gs_ratio * left + gs_ratio * right
        f_1 = sign * obj(x_1, *args)
        f_2 = sign * obj(x_2, *args)

        # Loop
        for _ in range(num_iterations):
            if f_1 > f_2:
                left = x_1
                x_1 = x_2
                f_1 = f_2
                x_2 = complementary_gs_ratio * left + gs_ratio * right
                f_2 = sign * obj(x_2, *args)
            else:
                right = x_2
                x_2 = x_1
                f_2 = f_1
                x_1 = gs_ratio * left + complementary_gs_ratio * right
                f_1 = sign * obj(x_1, *args)

        if f_1 < f_2:
            return x_1, sign * f_1
        return x_2, sign * f_2

    def _portfolio_return(self, weights):
        return self._as_scalar(weights.T @ self.expected_returns)

    def _portfolio_volatility(self, weights):
        variance = self._as_scalar(weights.T @ self.cov_matrix @ weights)
        return variance ** 0.5

    def _eval_sr(self, alpha, w_0, w_1):
        '''
        Evaluate the sharpe ratio of the portfolio within the convex combination

        :param alpha: (float) convex combination value
        :param w_0: (list) first endpoint of convex combination of weights
        :param w_1: (list) second endpoint of convex combination of weights
        :return:
        '''

        weights = alpha * w_0 + (1 - alpha) * w_1
        returns = self._portfolio_return(weights)
        volatility = self._portfolio_volatility(weights)
        return returns / volatility

    def _bound_free_weight(self, free_weights):
        '''
        Add a free weight to list of bounded weights

        :param free_weights: (list) list of free-weight indices
        :return: (float, int, int) lambda value, index of free weight to be bounded, bound weight value
        '''

        lambda_in = None
        i_in = None
        bi_in = None
        if len(free_weights) > 1:
            covar_f, covar_fb, mean_f, w_b = self._get_matrices(free_weights)
            covar_f_inv = np.linalg.inv(covar_f)
            for position, i in enumerate(free_weights):
                lambda_i, b_i = self._compute_lambda(
                    covar_f_inv,
                    covar_fb,
                    mean_f,
                    w_b,
                    position,
                    [self.lower_bounds[i], self.upper_bounds[i]],
                )
                if self._infnone(lambda_i) > self._infnone(lambda_in):
                    lambda_in, i_in, bi_in = lambda_i, i, b_i
        return lambda_in, i_in, bi_in

    def _free_bound_weight(self, free_weights):
        '''
        Add a bounded weight to list of free weights

        :param free_weights: (list) list of free-weight indices
        :return: (float, int) lambda value, index of the bounded weight to be made free
        '''

        lambda_out = None
        i_out = None
        if len(free_weights) < self.expected_returns.shape[0]:
            bounded_weight_indices = self._get_bounded_weights(free_weights)
            for i in bounded_weight_indices:
                covar_f, covar_fb, mean_f, w_b = self._get_matrices(free_weights + [i])
                covar_f_inv = np.linalg.inv(covar_f)
                lambda_i, _ = self._compute_lambda(
                    covar_f_inv,
                    covar_fb,
                    mean_f,
                    w_b,
                    mean_f.shape[0] - 1,
                    self.weights[-1][i],
                )
                if lambda_i is None:
                    continue
                if (self.lambdas[-1] is None or lambda_i < self.lambdas[-1]) and lambda_i > self._infnone(lambda_out):
                    lambda_out, i_out = lambda_i, i
        return lambda_out, i_out

    def _initialise(self, asset_prices, resample_by, covariance=None):
        '''
        Initialise covariances, upper-counds, lower-bounds and storage buffers

        :param asset_prices: (pd.Dataframe) dataframe of asset prices
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  'B' meaning daily business days which is equivalent to no resampling
        '''

        # Initial checks
        if not isinstance(asset_prices, pd.DataFrame):
            raise ValueError("Asset prices matrix must be a dataframe")
        if not isinstance(asset_prices.index, pd.DatetimeIndex):
            raise ValueError("Asset prices dataframe must be indexed by date.")

        # Resample the asset prices
        asset_prices = asset_prices.resample(resample_by).last()

        # Calculate the expected returns
        if self.calculate_returns == "mean":
            self.expected_returns = self._calculate_mean_historical_returns(asset_prices=asset_prices)
        elif self.calculate_returns == "exponential":
            self.expected_returns = self._calculate_exponential_historical_returns(asset_prices=asset_prices)
        else:
            raise ValueError(f"Unknown returns specified. Supported returns - {', '.join(sorted(SUPPORTED_RETURN_METHODS))}")
        self.expected_returns = self._to_column(self.expected_returns)
        if (self.expected_returns == np.ones(self.expected_returns.shape) * self.expected_returns.mean()).all():
            self.expected_returns[-1, 0] += 1e-5

        # Calculate the covariance matrix
        if covariance is None:
            self.cov_matrix = np.asarray(asset_prices.cov(), dtype=float)
        else:
            self.cov_matrix = np.asarray(covariance, dtype=float)

        # Intialise lower bounds
        if isinstance(self.weight_bounds[0], numbers.Real):
            self.lower_bounds = np.ones(self.expected_returns.shape) * self.weight_bounds[0]
        else:
            self.lower_bounds = self._to_column(self.weight_bounds[0])

        # Intialise upper bounds
        if isinstance(self.weight_bounds[0], numbers.Real):
            self.upper_bounds = np.ones(self.expected_returns.shape) * self.weight_bounds[1]
        else:
            self.upper_bounds = self._to_column(self.weight_bounds[1])
        if np.sum(self.lower_bounds) > 1 or np.sum(self.upper_bounds) < 1:
            raise ValueError("Weight bounds do not allow weights to sum to one.")

        # Initialise storage buffers
        self.weights = []
        self.lambdas = []
        self.gammas = []
        self.free_weights = []

    @staticmethod
    def _calculate_mean_historical_returns(asset_prices, frequency=252):
        '''
        Calculate the annualised mean historical returns from asset price data

        :param asset_prices: (pd.DataFrame) asset price data
        :return: (np.array) returns per asset
        '''

        returns = asset_prices.pct_change().dropna(how="all")
        returns = returns.mean() * frequency
        return returns

    @staticmethod
    def _calculate_exponential_historical_returns(asset_prices, frequency=252, span=500):
        '''
        Calculate the exponentially-weighted mean of (daily) historical returns, giving
        higher weight to more recent data.

        :param asset_prices: (pd.DataFrame) asset price data
        :return: (np.array) returns per asset
        '''

        returns = asset_prices.pct_change().dropna(how="all")
        returns = returns.ewm(span=span).mean().iloc[-1] * frequency
        return returns

    def allocate(self, asset_prices, solution="cla_turning_points", resample_by="B", covariance=None):
        # pylint: disable=consider-using-enumerate,too-many-locals,too-many-branches,too-many-statements
        '''
        Calculate the portfolio asset allocations using the method specified.

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (adj closed)
        :param solution: (str) specify the type of solution to compute. Options are: cla_turning_points, max_sharpe,
                               min_volatility, efficient_frontier
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  'B' meaning daily business days which is equivalent to no resampling
        :param covariance: (pd.Dataframe) a dataframe of covariance matrix (maybe shrinked or spectral rescaled)
        '''

        if solution not in SUPPORTED_SOLUTIONS:
            raise ValueError(
                "Unknown solution string specified. Supported solutions - "
                f"{', '.join(sorted(SUPPORTED_SOLUTIONS))}"
            )

        # Some initial steps before the algorithm runs
        self._initialise(asset_prices=asset_prices, resample_by=resample_by, covariance=covariance)
        assets = asset_prices.columns

        # Compute the turning points, free sets and weights
        free_weights, weights = self._init_algo()
        self.weights.append(np.copy(weights))  # store solution
        self.lambdas.append(None)
        self.gammas.append(None)
        self.free_weights.append(free_weights[:])
        while True:

            # 1) Bound one free weight
            lambda_in, i_in, bi_in = self._bound_free_weight(free_weights)

            # 2) Free one bounded weight
            lambda_out, i_out = self._free_bound_weight(free_weights)

            # 3) Compute minimum variance solution
            if (lambda_in is None or lambda_in < 0) and (lambda_out is None or lambda_out < 0):
                self.lambdas.append(0)
                covar_f, covar_fb, mean_f, w_b = self._get_matrices(free_weights)
                covar_f_inv = np.linalg.inv(covar_f)
                mean_f = np.zeros(mean_f.shape)

            # 4) Decide whether to free a bounded weight or bound a free weight
            else:
                if self._infnone(lambda_in) > self._infnone(lambda_out):
                    self.lambdas.append(lambda_in)
                    free_weights.remove(i_in)
                    weights[i_in] = bi_in  # set value at the correct boundary
                else:
                    self.lambdas.append(lambda_out)
                    free_weights.append(i_out)
                covar_f, covar_fb, mean_f, w_b = self._get_matrices(free_weights)
                covar_f_inv = np.linalg.inv(covar_f)

            # 5) Compute solution vector
            w_f, gamma = self._compute_w(covar_f_inv, covar_fb, mean_f, w_b)
            for i in range(len(free_weights)):
                weights[free_weights[i]] = w_f[i]
            self.weights.append(np.copy(weights))  # store solution
            self.gammas.append(gamma)
            self.free_weights.append(free_weights[:])
            if self.lambdas[-1] == 0:
                break

        # 6) Purge turning points
        self._purge_num_err(10e-10)
        self._purge_excess()

        # Compute the specified solution
        self._compute_solution(assets=assets, solution=solution)

    def _compute_solution(self, assets, solution):
        '''
        Compute the desired solution to the portfolio optimisation problem

        :param assets: (list) a list of asset names
        :param solution: (str) specify the type of solution to compute. Options are: cla_turning_points, max_sharpe,
                               min_volatility, efficient_frontier
        '''

        if solution == "max_sharpe":
            self.max_sharpe, self.weights = self._max_sharpe()
            self.weights = self._weights_to_frame([self.weights], assets)
        elif solution == "min_volatility":
            self.min_var, self.weights = self._min_volatility()
            self.weights = self._weights_to_frame([self.weights], assets)
        elif solution == "efficient_frontier":
            self.efficient_frontier_means, self.efficient_frontier_sigma, self.weights = self._efficient_frontier()
            self.weights = self._weights_to_frame(self.weights, assets)
        elif solution == "cla_turning_points":
            self.weights = self._weights_to_frame(self.weights, assets)

    def _max_sharpe(self):
        '''
        Compute the maximum sharpe portfolio allocation

        :return: (float, np.array) tuple of max. sharpe value and the set of weight allocations
        '''

        # 1) Compute the local max SR portfolio between any two neighbor turning points
        w_sr, sharpe_ratios = [], []
        if len(self.weights) < 2:
            raise ValueError("At least two CLA turning points are required to compute maximum Sharpe.")

        for i in range(len(self.weights) - 1):
            w_0 = np.copy(self.weights[i])
            w_1 = np.copy(self.weights[i + 1])
            kwargs = {"minimum": False, "args": (w_0, w_1)}
            alpha, sharpe_ratio = self._golden_section(self._eval_sr, 0, 1, **kwargs)
            w_sr.append(alpha * w_0 + (1 - alpha) * w_1)
            sharpe_ratios.append(sharpe_ratio)

        maximum_sharp_ratio = max(sharpe_ratios)
        weights_with_max_sharpe_ratio = w_sr[sharpe_ratios.index(maximum_sharp_ratio)]
        return maximum_sharp_ratio, weights_with_max_sharpe_ratio

    def _min_volatility(self):
        '''
        Compute minimum volatility portfolio allocation

        :return: (float, np.array) tuple of minimum variance value and the set of weight allocations
        '''

        var = []
        for weights in self.weights:
            volatility = self._as_scalar(weights.T @ self.cov_matrix @ weights)
            var.append(volatility)
        min_var = min(var)
        return min_var ** .5, self.weights[var.index(min_var)]

    def _efficient_frontier(self, points=100):
        # pylint: disable=invalid-name
        '''
        Compute the entire efficient frontier solution

        :param points: (int) number of efficient frontier points to be calculated
        :return: tuple of mean, variance amd weights of the frontier solutions
        '''

        means, sigma, weights = [], [], []
        if len(self.weights) < 2:
            return means, sigma, weights

        points_per_segment = max(points // len(self.weights), 1)
        segment_indices = range(len(self.weights) - 1)
        default_partitions = np.linspace(0, 1, points_per_segment)[:-1]

        for i in segment_indices:
            w_0, w_1 = self.weights[i], self.weights[i + 1]
            if i == len(self.weights) - 2:
                partitions = np.linspace(0, 1, points_per_segment)
            else:
                partitions = default_partitions

            for partition in partitions:
                w = w_1 * partition + (1 - partition) * w_0
                weights.append(np.copy(w))
                means.append(self._portfolio_return(w))
                sigma.append(self._portfolio_volatility(w))
        return means, sigma, weights
