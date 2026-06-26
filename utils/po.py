# from https://github.com/hudson-and-thames/mlfinlab
# license: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

'''
This module implements the classic mean-variance optimization techniques for calculating the efficient frontier.
It uses typical quadratic optimizers to generate optimal portfolios for different objective functions.
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimization:
    '''
    This class contains a variety of methods dealing with different solutions to the mean variance optimization
    problem.
    '''

    def __init__(self):
        self.weights = list()

    def allocate(self, asset_prices, solution='inverse_variance', resample_by='B', covariance=None):
        '''
        Calculate the portfolio asset allocations using the method specified.

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param solution: (str) the type of solution/algorithm to use to calculate the weights
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  'B' meaning daily business days which is equivalent to no resampling
        :param covariance: (pd.Dataframe) a dataframe of covariance matrix (maybe shrinked or spectral rescaled)
        '''

        if not isinstance(asset_prices, pd.DataFrame):
            raise ValueError("Asset prices matrix must be a dataframe")
        if not isinstance(asset_prices.index, pd.DatetimeIndex):
            raise ValueError("Asset prices dataframe must be indexed by date.")

        # Calculate returns
        asset_returns = self._calculate_returns(asset_prices, resample_by=resample_by)
        assets = asset_prices.columns

        if solution == 'inverse_variance':
            if covariance is None:
                self.weights = self._inverse_variance(covariance=asset_returns.cov())
            else:
                self.weights = self._inverse_variance(covariance=covariance)
        elif solution == 'equal_weight':
            self.weights = self._equal_weight(assets=assets)
        elif solution == "mean_variance":
            self.weights = self._mean_variance(asset_returns=asset_returns)
        else:
            raise ValueError("Unknown solution string specified. Supported solutions - inverse_variance, equal_weight, mean_variance.")

        self.weights = pd.DataFrame(self.weights)
        self.weights.index = assets
        self.weights = self.weights.T

    @staticmethod
    def _calculate_returns(asset_prices, resample_by):
        '''
        Calculate the annualised mean historical returns from asset price data

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  'B' meaning daily business days which is equivalent to no resampling
        :return: (pd.Dataframe) stock returns
        '''

        asset_prices = asset_prices.resample(resample_by).last()
        asset_returns = asset_prices.pct_change()
        asset_returns = asset_returns.dropna(how='all')
        return asset_returns

    @staticmethod
    def _inverse_variance(covariance):
        '''
        Calculate weights using inverse-variance allocation

        :param covariance: (pd.Dataframe) covariance dataframe of asset returns
        :return: (np.array) array of portfolio weights
        '''

        ivp = 1. / np.diag(covariance)
        ivp /= ivp.sum()
        return ivp

    @staticmethod
    def _mean_variance(asset_returns):
        '''
        Calculate weights using long-only mean-variance allocation.

        The objective is the maximum Sharpe/tangency portfolio with a zero
        risk-free rate, constrained so weights sum to one and remain in [0, 1].

        :param asset_returns: (pd.Dataframe) asset returns
        :return: (np.array) array of portfolio weights
        '''

        returns = asset_returns.dropna(how='any')
        if returns.empty:
            raise ValueError("Asset returns dataframe must contain at least one complete return observation.")

        expected_returns = returns.mean().values
        covariance = returns.cov().values
        num_assets = len(expected_returns)
        if num_assets == 0:
            raise ValueError("Asset returns dataframe must contain at least one asset.")

        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(covariance, weights))
            if portfolio_variance <= 0:
                return np.inf
            return -portfolio_return / np.sqrt(portfolio_variance)

        initial_weights = np.ones(num_assets) / num_assets
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0},)
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))

        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )
        if not result.success:
            raise ValueError(f"Mean-variance optimization failed: {result.message}")

        weights = np.clip(result.x, 0.0, 1.0)
        return weights / weights.sum()

    @staticmethod
    def _equal_weight(assets):
        '''
        Calculate weights using equal-weight allocation

        :param assets: (list) list of asset names
        :return: (np.array) array of portfolio weights
        '''

        return np.ones(len(assets)) / len(assets)
