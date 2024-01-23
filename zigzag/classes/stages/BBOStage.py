from typing import Generator
import numpy as np

from zigzag.classes.stages.Stage import Stage
from zigzag.classes.opt.NDO.bayes_opt import BayesianOptimizer

import logging
from termcolor import cprint

logger = logging.getLogger(__name__)

# Non-Differentiable Optimizer stage
class BBOStage(Stage):
    def __init__(self, list_of_callables, *, optimizer_params, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.optimizer_params = optimizer_params
        self.kwargs = kwargs
        
        
    def run(self) -> Generator:
        if self.optimizer_params['optimizer_type'] == 'bayesian_optimization':
            opt_engine = BayesianOptimizer(self.list_of_callables, self.kwargs, self.optimizer_params)
            opt_engine.runner()
       
if __name__ == "__main__":
    pass
