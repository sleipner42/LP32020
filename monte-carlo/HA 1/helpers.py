import pandas as pd
import numpy as np
import math
from scipy.stats import weibull_min, invweibull, uniform, norm
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class helper_methods:

    months = {
        "jan" : [10.6, 2.0, 12, 4.3],
        "feb" : [9.7, 2.0, 12, 4.3],
        "mar" : [9.2, 2.0, 12, 4.3],
        "apr" : [8.0, 1.9, 12, 4.3],
        "may" : [7.8, 1.9, 12, 4.3],
        "jun" : [8.1, 1.9, 12, 4.3],
        "jul" : [7.8, 1.9, 12, 4.3],
        "aug" : [8.1, 1.9, 12, 4.3],
        "sep" : [9.1, 2.0, 12, 4.3],
        "oct" : [9.9, 1.9, 12, 4.3],
        "nov" : [10.6, 2.0, 12, 4.3],
        "dec" : [10.6, 2.0, 12, 4.3]
    }

    winds = {}

    def __init__(self, n):
        super().__init__()
        self.n = n

    def init_winds(self):
        
        for key, val in self.months.items():

            lam = val[0]
            k = val[1]

            # Creating the wind distribution for a month
            wind_distribution =  weibull_min(k, loc=0, scale=lam)

            # Generate wind
            wind = wind_distribution.rvs(size=self.n)

            self.winds[key] = {"wind" : wind, "dist" : wind_distribution}

    def importance_sampling(self, f, phi_dist, g_dist):

        x = np.linspace(0, 30, num=self.n)

        # Sampling from g
        X = g_dist.rvs(size=self.n)

        # Function to evaluate
        s = lambda x: phi_dist.pdf(x)*f(x)/g_dist.pdf(x)
        
        sample = s(X)
        
        # Calculate variance
        mean = np.mean(sample)
        var = np.var(sample)

        confidence_interval = 1.96*np.sqrt(var/self.n)

        return {"mean" : mean, "var": var, "ci" : confidence_interval}