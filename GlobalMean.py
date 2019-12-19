from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import AlgoBase


class GlobalMean(AlgoBase):

    def __init__(self, bsl_options={}, verbose=True):
        AlgoBase.__init__(self, bsl_options=bsl_options)
        self.verbose = verbose
        
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        bu,bi = self.compute_baselines() 
        return self

    def estimate(self, u, i): 
        return self.default_prediction()
