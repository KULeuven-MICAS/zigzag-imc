from zigzag.classes.stages.Stage import Stage
from zigzag.classes.hardware.architecture.ImcArray import ImcArray
from zigzag.classees.opt.NDO.black_box_optimizer import BlackBoxOptimizer, tID, OptimizerTarget
from zigzag.classes.opt.NDO.utils import imc_array_dut
# Bayesian optimization imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import numpy as np
import pickle
import logging

logger = logging.getLogger(__name__)


class BayesianOptimizer(BlackBoxOptimizer):
    def __init__(self, optimizer_params):
        self.opt_targets = optimizer_params['optimizer_targets']
        self.input_range = np.array([np.array(x) for x in itertools.product(*[t.target_range for t in self.opt_targets])])
        self.input_samples = []
        self.output_samples = []
        self.best_cost = -float('inf')
        self.optimizer_params = optimizer_params
    
    def run(self):
        self.init_optimizer()
        for i in range(self.optimizer_params['iterations']):
            optimizer_target = self.sample(ei=self.ei) 
            self.get_cost(optimizer_target)
            self.update_optimizer()
        with open('data.pkl','wb') as infile:
            pickle.dump([self.input_samples, self.output_samples], infile)
        breakpoint()

    def sample(self, ei=None, init=False):
        while(1):
            if init:
                dims = np.random.choice(self.input_range)
            else:
                dims = self.input_range[np.argmax(ei)]
            dimensions = {'D1':int(dims[0]),'D2':int(dims[1]),'D3':1}
            sample_correct = self.check_sample_correctness(dimensions)
            if sample_correct:
                ba_mask = []
                for ii_xx, xx in enumerate(self.input_range):
                    if xx[0] == dims[0] and xx[1] == dims[1]:
                        ba_mask.append(ii_xx)
                self.input_range = np.delete(self.input_range, ba_mask, axis=0)
                if not init:
                    self.ei = np.delete(self.ei, ba_mask, axis=0)
                break
            else:
                if not init:
                    ba_mask = []
                    print(f'Unfit dims {dimensions}')
                    for ii_xx, xx in enumerate(self.input_range):
                        if xx[0] >= dims[0] and xx[1] >= dims[1]:
                            ba_mask.append(ii_xx)
                    self.input_range = np.delete(self.input_range, ba_mask, axis=0)
                    self.ei = np.delete(self.ei, ba_mask, axis=0)

        self.input_samples.append(dims)
        for ii_ot, optimizer_target in enumerate(self.optimizer_targets):
            optimizer_target.target_parameters = dims[ii_ot]

    
    def check_sample_correctness(self, dimensions):
        group_depth = 1
        imc_array = imc_array_dut(dimensions, group_depth)
        if imc_array.total_area <= self.optimizer_params['area_budget']:
            return True
        else:
            return False
            
    def init_optimizer(self):
        for i in range(self.optimizer_params['init_iterations']):
            optimizer_target = self.sample(init=True)
            self.get_cost(optimizer_target)
        self.init_surrogate()
        self.update_optimizer()

    def get_cost(self):
        self.kwargs['optimizer_target'] = optimizer_target 
        logger.info(f'Search array dimension: {[optimizer_target.target_parameters["D1"], optimizer_target.target_parameters["D2"]]}')
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:],**self.kwargs)
        cme_list = []
        for cme, extra_info in sub_stage.run():
            cme_list.append(cme)
            yield cme, extra_info
        cost = self.get_cost(cme_list)
        if cost > self.best_cost:
            self.best_cost = cost
        logger.info(f'Cost {cost:6.2e} Best cost {self.best_cost:6.2e}')
        self.output_samples.append(cost)

    def update_optimizer(self):
        self.gp_model.fit(np.array(self.input_samples), np.array(self.output_samples))
        self.acquisition_function()

    def init_surrogate(self):
        self.kernel = RBF(length_scale=1.0)
        self.gp_model = GaussianProcessRegressor(kernel=self.kernel)
        self.gp_model.fit(np.array(self.input_samples), np.array(self.output_samples))
        self.pred, self.std = self.gp_model.predict(self.input_range, return_std=True)

    def acquisition_function(self):
        def expected_improvement(x, gp_model, best_y):
            y_pred, y_std = gp_model.predict(x, return_std=True)
            z = (y_pred - best_y) / y_std
            ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
            return ei
        # Determine the point with the highest observed function value
        best_idx = np.argmax(np.array(self.bo_output_samples))
        best_y = np.array(self.bo_output_samples)[best_idx]
        self.ei = expected_improvement(np.array(self.bo_input_range), self.gp_model, best_y)

