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

# Non-Differentiable Optimizer stage
class GDStage(Stage):
    def __init__(self, list_of_callables, *, optimizer_params, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.optimizer_params = optimizer_params
        self.batch_size = 1
        self.epochs = 20
        self.l1 = 1

    def run(self) -> Generator:

        # Find random initial point
        while(1):
            dims = np.random.choice(np.arange(32,512+1,32), size=2)
            dimensions = {'D1':int(dims[0]),'D2':int(dims[1]),'D3':1}
            group_depth = 1
            imc_array = self.imc_array_dut(dimensions, group_depth)
            if imc_array.total_area <= self.optimizer_params['area_budget']:
                break

        opt_dimension = [dimensions]        
        cost_list = []
        for i in range(self.epochs):
            gradient_list = []
            optimizer_target = OptimizerTarget(target_stage = 'AcceleratorParserStage',
                            target_object = [tID('object','accelerator'), tID('obj','cores'), tID('list',0), tID('obj','operational_array')],
                            target_modifier = 'set_array_dim',
                            target_parameters = dimensions)
            c_x = self.get_cost(optimizer_target)
            cost_list.append(c_x)
            cprint(f'EPOCH {i} Dims: {dimensions} Cost: {c_x:.2e}','green')
            for b in range(self.batch_size):
                # create perturbation vector
                while(1):
                    fit = True
                    perturbation_vector = np.random.normal(0,64, size=2)
                    for ii_dim, dim in enumerate(['D1','D2']):
                        for ii_v, v in enumerate(perturbation_vector):
                            temp_dimensions = dimensions.copy()
                            temp_dimensions[dim] = max(64,int(temp_dimensions[dim] + v))
                            imc_array = self.imc_array_dut(temp_dimensions, group_depth)
                            if imc_array.total_area > self.optimizer_params['area_budget']:
                                fit == False
                                break
                        if fit == False:
                            break
                    if fit == True:
                        break
                #compute jvp per dimension, per perturbation vector
                gradient_dict = np.zeros((2,2))
                for ii_dim, dim in enumerate(['D1','D2']):
                    for ii_v, v in enumerate(perturbation_vector):
                        # update parameters in model temporarily
                        temp_dimensions = dimensions.copy()
                        #temp_dimensions[dim] = int(np.exp(np.log(temp_dimensions[dim]) + v))
                        temp_dimensions[dim] = max(32,int(temp_dimensions[dim] + v))
                        optimizer_target = OptimizerTarget(target_stage = 'AcceleratorParserStage',
                            target_object = [tID('object','accelerator'), tID('obj','cores'), tID('list',0), tID('obj','operational_array')],
                            target_modifier = 'set_array_dim',
                            target_parameters = temp_dimensions)
                        # compute cost function in perturbed point
                        c = self.get_cost(optimizer_target)
                        # numerical differentiation
                        n_diff = (c - c_x) / c_x
                        # update jvp matrix
                        gradient_dict[ii_dim, ii_v] = n_diff * perturbation_vector[ii_dim]
                        print(f'Batch {b}: Grad {dim}: Original: {dimensions} --> {temp_dimensions}: Cost {c:.2e}, gradient: {gradient_dict[ii_dim,ii_v]:.2e}')
                print(gradient_dict)
                gradient = np.sum(gradient_dict, axis=1)
                gradient_list.append(gradient)
            # average out gradients
            avg_gradient = np.mean(np.array(gradient_list),axis=0)
            print('AVG gradient', avg_gradient)
            # update parameters
            new_dimensions = dimensions.copy()
            for ii_dim, dim in enumerate(['D1','D2']):
                new_dimensions[dim] = max(32, int(dimensions[dim] - self.l1 * avg_gradient[ii_dim]))
            imc_array = self.imc_array_dut(new_dimensions, group_depth)
            if imc_array.total_area <= self.optimizer_params['area_budget']:
                dimensions = new_dimensions.copy()

            opt_dimension.append(dimensions.copy())


        import pickle
        print(opt_dimension)
        with open('data.pkl','wb') as infile:
            pickle.dump([cost_list, opt_dimension], infile)

        breakpoint() 


    def get_cost(self, optimizer_target):
        self.kwargs['optimizer_target'] = optimizer_target 
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:],**self.kwargs)
        cme_list = []
        for cme, extra_info in sub_stage.run():
            cme_list.append(cme)
    #        yield cme, extra_info
        return sum([x.energy_total * x.latency_total0 for x in cme_list])


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
