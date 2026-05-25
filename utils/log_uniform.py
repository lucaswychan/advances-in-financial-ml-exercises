import numpy as np
from scipy.stats import rv_continuous

class LogUniform(rv_continuous):
    # random numbers log-uniformly distributed between 1 and e
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)
    
def log_uniform(a=1, b=np.exp(1)):
    return LogUniform(a=a, b=b, name='LogUniform')