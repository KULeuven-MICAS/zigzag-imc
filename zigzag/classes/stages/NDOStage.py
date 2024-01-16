from typing import Generator
import numpy as np

from zigzag.classes.stages.Stage import Stage
from zigzag.classes.hardware.architecture.ImcArray import ImcArray

# Bayesian optimization imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

import logging
from termcolor import cprint

logger = logging.getLogger(__name__)

class tID():
    def __init__(self, target_type, target_id):
        self.target_type = target_type
        self.target_id = target_id

class OptimizerTarget():
    def __init__(self, target_stage, target_object, target_modifier, target_parameters):
        self.target_stage = target_stage
        self.target_object = target_object
        self.target_modifier = target_modifier
        self.target_parameters = target_parameters

# Non-Differentiable Optimizer stage
class NDOStage(Stage):
    def __init__(self, list_of_callables, *, optimizer_params, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.optimizer_params = optimizer_params
        
    def run(self) -> Generator:


        # Simulated annealing
        if self.optimizer_params['optimizer_type'] == 'simulated_annealing':
            x = np.linspace(32,256,256-32+1)
            self.sa_input_samples = []
            self.sa_output_samples = []
            self.sa_temp = self.optimizer_params['init_temperature']
            self.sa_alfa = self.optimizer_params['cooling_factor']

            # Sample new point
            optimizer_target = self.sa_sample_parameter(init=True)
            self.kwargs['optimizer_target'] = optimizer_target 
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:],**self.kwargs)
            cme_list = []
            for cme, extra_info in sub_stage.run():
                cme_list.append(cme)
                yield cme, extra_info
            self.sa_best_cost = - self.get_cost(cme_list)
            self.sa_output_samples.append(self.sa_best_cost)
            current_dim = [optimizer_target.target_parameters['D1'], optimizer_target.target_parameters['D2']]

            print(f'Cost {self.sa_best_cost:.2e}')
            for i in range(self.optimizer_params['iterations']):
                # Sample new point from distribution g(_) ==> random perturbation
                optimizer_target = self.sa_sample_parameter(init=False, current_dim=current_dim)
                self.kwargs['optimizer_target'] = optimizer_target 
                sub_stage = self.list_of_callables[0](self.list_of_callables[1:],**self.kwargs)
                cme_list = []
                for cme, extra_info in sub_stage.run():
                    cme_list.append(cme)
                    yield cme, extra_info
                new_cost = - self.get_cost(cme_list)
                print(f'Cost {new_cost:.2e}')
                self.sa_output_samples.append(new_cost)
                delta_cost = new_cost - self.sa_best_cost
                print(f'{delta_cost:.2e}', np.log10(delta_cost))
                # Apply acceptance criterion
                if (delta_cost < 0):
                    #accept
                    cprint('Accepted because negative', 'green')
                    self.sa_best_cost = new_cost
                    current_dim = [optimizer_target.target_parameters['D1'], optimizer_target.target_parameters['D2']]
                else:
                    r = np.random.random()
                    cprint(f'Accept? {r:.2e} < {np.exp(- np.log(delta_cost)/10 / self.sa_temp):.2e}','yellow')
                    if r < np.exp(- np.log(delta_cost)/10 / self.sa_temp):
                        #accept
                        cprint('Accepted','green')
                        self.sa_best_cost = new_cost
                        current_dim = [optimizer_target.target_parameters['D1'], optimizer_target.target_parameters['D2']]
                    else:
                        # not accept
                        cprint('Refused','red')
                        pass

                print()
                # update temperature
                # Geometric cooling adopted (Also logarithmic, exponential available)
                #self.sa_temp = np.power(self.sa_alfa,i) * self.optimizer_params['init_temperature']
                # Logarithmic cooling
                self.sa_temp = (self.sa_alfa * self.optimizer_params['init_temperature']) / (np.log(2+i))
                print(f'Iteration {i}: Temperature {self.sa_temp:.2e}')
                
            breakpoint()
            
        # Bayesian optimization
        if self.optimizer_params['optimizer_type'] == 'bayesian_optimization':
            self.bo_input_samples = []
            self.bo_output_samples = np.array([])
            self.bo_input_range = np.dstack(np.meshgrid(x,x)).reshape(-1,2)

            # Initialize surrogate model with set of samples
            for i in range(self.optimizer_params['init_iterations']):
                optimizer_target = self.bo_sample_parameter(init=True)
                self.kwargs['optimizer_target'] = optimizer_target 
                sub_stage = self.list_of_callables[0](self.list_of_callables[1:],**self.kwargs)
                cme_list = []
                for cme, extra_info in sub_stage.run():
                    cme_list.append(cme)
                    yield cme, extra_info
                cost = self.get_cost(cme_list)
                self.bo_output_samples = np.append(self.bo_output_samples, cost)

            # Initialize surrogate function (duh)
            self.bo_init_surrogate()
            # Find next point(s) to be sampled with EI
            ei = self.bo_acquisition_function()
            for i in range(self.optimizer_params['iterations']):
                optimizer_target = self.bo_sample_parameter(ei=ei) 
                self.kwargs['optimizer_target'] = optimizer_target 
                sub_stage = self.list_of_callables[0](self.list_of_callables[1:],**self.kwargs)
                cme_list = []
                for cme, extra_info in sub_stage.run():
                    cme_list.append(cme)
                    yield cme, extra_info
                cost = self.get_cost(cme_list)
                self.bo_output_samples = np.append(self.bo_output_samples, cost)
                ei = self.bo_acquisition_function()

            best_parameters = self.bo_get_best_point()
            print(self.bo_output_samples)
            print(best_parameters)
            breakpoint()


    def sa_sample_parameter(self, init=False, current_dim=None):
        while(1):
            if init:
                dims = np.random.randint(32, 256, size=2)
            else:
                while(1):
                    dims = [x for x in current_dim]
                    dims[0] = current_dim[0] + int( np.random.normal(scale=self.sa_temp)*256 )
                    dims[1] = current_dim[1] + int( np.random.normal(scale=self.sa_temp)*256 )
                    if dims[0] > 0 and dims[1] > 0:
                        break

            dimensions = {'D1':int(dims[0]),'D2':int(dims[1]),'D3':1}
            group_depth = 1
            imc_array = self.imc_array_dut(dimensions, group_depth)
            if imc_array.total_area <= self.optimizer_params['area_budget']:
                break

        self.sa_input_samples.append(dims)
        print(f'Sampled array dimension: {dims}')
        optimizer_target = OptimizerTarget(target_stage = 'AcceleratorParserStage',
                target_object = [tID('object','accelerator'), tID('obj','cores'), tID('list',0), tID('obj','operational_array')],
                target_modifier = 'set_array_dim',
                target_parameters = dimensions)
        return optimizer_target


    def bo_sample_parameter(self, ei=None, init=False):
        while(1):
            if init:
                dims = np.random.randint(32, 256, size=2)
            else:
                dims = self.bo_input_range[np.argmax(ei)]
            dimensions = {'D1':int(dims[0]),'D2':int(dims[1]),'D3':1}
            group_depth = 1
            imc_array = self.imc_array_dut(dimensions, group_depth)
            print(dims, imc_array.total_area)
            if imc_array.total_area <= self.optimizer_params['area_budget']:
                break
            else:
                if not init:
                    self.bo_input_range = np.delete(self.bo_input_range, np.where(self.bo_input_range==dims))
                    ei = np.delete(ei, np.where(ei==np.amax(ei)))

        self.bo_input_samples.append(dims)
        print(dims)
        optimizer_target = OptimizerTarget(target_stage = 'AcceleratorParserStage',
                target_object = [tID('object','accelerator'), tID('obj','cores'), tID('list',0), tID('obj','operational_array')],
                target_modifier = 'set_array_dim',
                target_parameters = dimensions)
        return optimizer_target

    def get_cost(self, cme_list):
        return -sum([x.energy_total * x.latency_total0 for x in cme_list])

    def bo_init_surrogate(self):
        self.bo_kernel = RBF(length_scale=1.0)
        self.gp_model = GaussianProcessRegressor(kernel=self.bo_kernel)
        self.gp_model.fit(np.array(self.bo_input_samples), self.bo_output_samples)
        self.bo_pred, self.bo_std = self.gp_model.predict(self.bo_input_range, return_std=True)

    def bo_acquisition_function(self):
        def expected_improvement(x, gp_model, best_y):
            y_pred, y_std = gp_model.predict(x, return_std=True)
            z = (y_pred - best_y) / y_std
            ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
            return ei
        # Determine the point with the highest observed function value
        self.gp_model.fit(np.array(self.bo_input_samples), self.bo_output_samples)
        best_idx = np.argmax(self.bo_output_samples)
        #best_x = self.bo_input_samples[best_idx]
        best_y = self.bo_output_samples[best_idx]
        ei = expected_improvement(self.bo_input_range, self.gp_model, best_y)
        return ei

    def bo_get_best_point(self):
        best_idx = np.argmax(self.bo_output_samples)
        best_y = self.bo_output_samples[best_idx]
        return best_y

    def imc_array_dut(self, dimensions, group_depth):
        """Multiplier array variables"""
        tech_param = { # 28nm
            "tech_node": 0.028,             # unit: um
            "vdd":      0.9,                # unit: V
            "nd2_cap":  0.7/1e3,            # unit: pF
            "xor2_cap": 0.7*1.5/1e3,        # unit: pF
            "dff_cap":  0.7*3/1e3,          # unit: pF
            "nd2_area": 0.614/1e6,          # unit: mm^2
            "xor2_area":0.614*2.4/1e6,      # unit: mm^2
            "dff_area": 0.614*6/1e6,        # unit: mm^2
            "nd2_dly":  0.0478,             # unit: ns
            "xor2_dly": 0.0478*2.4,         # unit: ns
            # "dff_dly":  0.0478*3.4,         # unit: ns
        }
        hd_param = {
            "pe_type":              "in_sram_computing",     # for in-memory-computing. Digital core for different values.
            "imc_type":             "digital",  # "digital" or "analog"
            "input_precision":      8,          # activation precision expected in the hardware
            "weight_precision":     8,          # weight precision expected in the hardware
            "input_bit_per_cycle":  1,          # nb_bits of input/cycle/PE
            "group_depth":          group_depth,          # group depth in each PE
            "wordline_dimension": "D1",         # hardware dimension where wordline is (corresponds to the served dimension of input regs)
            "bitline_dimension": "D2",          # hardware dimension where bitline is (corresponds to the served dimension of output regs)
            "enable_cacti":         True,       # use CACTI to estimated cell array area cost (cell array exclude build-in logic part)
            "adc_resolution": max(1, 0.5 * np.log2(dimensions['D1']))
            # Energy of writing weight. Required when enable_cacti is False.
            # "w_cost_per_weight_writing": 0.08,  # [OPTIONAL] unit: pJ/weight.
        }

        imc_array = ImcArray(
            tech_param, hd_param, dimensions
        )

        return imc_array


    # # For testing purposes
    # def is_leaf(self) -> bool:
    #     return True
