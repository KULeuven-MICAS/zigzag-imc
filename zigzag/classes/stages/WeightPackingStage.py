import networkx as nx
import logging
import numpy as np
import itertools
import copy
from rectpack import newPacker
import rectpack.packer as packer
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ortools.sat.python import cp_model
from loguru import logger

from zigzag.classes.stages.Stage import Stage
from zigzag.classes.workload.dummy_node import DummyNode
from zigzag.classes.opt.CGA.item import Item, ItemPool
from zigzag.classes.opt.CGA.superitem import *
from zigzag.classes.opt.CGA.macro_bin import MacroBin
from zigzag.classes.opt.CGA.layer import *
from zigzag.classes.opt.CGA.utils import plot_item_allocation, prime_factors



logger = logging.getLogger(__name__)

class WeightPackingStage(Stage):

    ## The class constructor
    # Initialization of self.workload.
    def __init__(self, list_of_callables, *, workload, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload

#    def weight_reloading_cost(self, layer_list, not_allocated_layers, bin_dict):
#        not_allocated_layers.sort(key=lambda x:x.height, reverse=True)
#        not_allocated_layers_height = [x.height for x in 

    def weight_tile_allocation(self, ox_unrolling_scheme):
        network = self.extract_network_from_workload()
        solver_status = ""
        D1, D2, D3, M = self.get_IMC_dimension_parameters() 
        kwargs = self.kwargs.copy()

        itempool = ItemPool(D1=D1,D2=D2,D3=D3,M=M,network=network, ox_unrolling_scheme=ox_unrolling_scheme)
        itempool.set_init_M()
        oxu_scheme_str =[f"L{x[0]} OXu {x[1]}" for x in ox_unrolling_scheme] 
        logger.info(f'OX unrolling scheme {oxu_scheme_str}')
        while solver_status not in ['OPTIMAL','FEASIBLE']:
        #    logger.info("===== ItemPool Generation =====")
            item_pool, feasible_tile_configuration, target_layer_index = itempool.generate()
            while not feasible_tile_configuration:
                feasible_tile_configuration = itempool.update_mapping(target_layer_index)
                if not feasible_tile_configuration:
                    break
                item_pool, feasible_tile_configuration, target_layer_index = itempool.generate()
            if not feasible_tile_configuration:
                logger.error('>>>> Unfeasible settings for D1,D2,D3,M <<<<')
                return [], False
            si = SuperItemPool(item_pool)
        #    logger.info("===== SuperItemPool Generation =====")
            superitem_pool = si.generate()
        #    logger.info("===== LayerPool Generation =====")
            layer_pool = LayerPool(D1=D1,D2=D2, network_layers=len(network.keys()))
            layer_list = layer_pool.generate(superitem_pool)
#            fsi, zsl, ol, nki = layer_pool.generate_cp_parameters()
        #    logger.info("===== Bin Allocation =====")
            macro_bin = MacroBin(height=M, number_of_macros=D3)
            bin_dict, solver_status, not_allocated_layers = macro_bin.macro_allocation(layer_list)
            #bin_dict, solver_status = macro_bin.pack_macrobin(layer_list, fsi, zsl, ol, nki)
            if solver_status in ['OPTIMAL','FEASIBLE']:
        #        plot_item_allocation(layer_list, bin_dict, D3=int(D3), height=int(M), D1=int(D1),D2=int(D2))
 #               self.generate_mappings(network, item_pool)
                utilization = self.get_utilization(item_pool) / D1 / D2 / D3 / M
                logger.info(f'>>>> Completed allocation <<<< Utilization {utilization*100:.2f}%')
                break
            else:
                # Estimate cost of weight reloading
                self.weight_reloading_cost(layer_list, not_allocated_layers, bin_dict)
                # Update Mapping
                feasible_configuration = itempool.update_mapping()
            if not feasible_configuration:
                logger.error('>>>> Unfeasible settings for D1,D2,D3,M <<<<')
                return [], False
        kwargs['workload'] = self.workload
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        cme_list = []
        for cme, (layer, extra_info) in sub_stage.run():
            cme_list.append(cme)
        cost = sum([x.energy_total * x.latency_total0 for x in cme_list])
        logger.info(f'Allocation cost: {cost:.2e}')
        return cost, True

    def get_utilization(self, item_pool):
        return sum([x.volume * x.tile_index for x in item_pool])
    
    def run(self):

        network = self.extract_network_from_workload()
        valid_ox_unrolling_scheme = []
        cme_list = []
        min_cost = float('inf')
        for layer_index, layer in network.items():
            ox_pf = [x for x in prime_factors(layer['OX'])]
            ox_comb = []
            for k in range(1,len(ox_pf)+1):
                for comb in itertools.combinations(ox_pf, k):
                    if np.prod(comb) not in ox_comb:
                        ox_comb.append(np.prod(comb))
            if layer_index == 0:
                ox_comb.insert(0,1)
            ox_comb.sort()
            ox_comb = [(layer_index, x) for x in ox_comb] 
            if valid_ox_unrolling_scheme == []:
                ox_unrolling_scheme_list = [[x] for x in ox_comb]
            else:
                ox_unrolling_scheme_list = itertools.product(*[ox_comb, valid_ox_unrolling_scheme])
                ox_unrolling_scheme_list = [[x[0]] + x[1] for x in ox_unrolling_scheme_list]
            for ox_unrolling_scheme in ox_unrolling_scheme_list:
                cme, feasible = self.weight_tile_allocation(ox_unrolling_scheme)
                if not feasible:
                    break
                if cme > min_cost:
                    break
                min_cost = cme
                valid_ox_unrolling_scheme.append(ox_unrolling_scheme)
                cme_list.append((ox_unrolling_scheme,cme))

        breakpoint()
        yield 0,0


    def generate_mappings(self, network, item_pool):
        for layer in self.workload:
            if type(layer) == DummyNode:
                continue  # skip the DummyNodes

            layer_id = next((k for k,v in network.items() if v['layer_id'] == layer.id),None)
            layer_item = next((x for x in item_pool if x.layer_index == layer_id),None) 
            new_spatial_mapping = {'D1':layer_item.D1_unroll, 'D2':layer_item.D2_unroll, 'D3':layer_item.D3_unroll}
            layer.user_spatial_mapping = new_spatial_mapping


    def get_IMC_dimension_parameters(self):
        imc_dim = self.kwargs['accelerator'].cores[0].operational_array.dimensions
        D1 = next((x for x in imc_dim if x.name == 'D1'),None).size
        D2 = next((x for x in imc_dim if x.name == 'D2'),None).size
        D3 = next((x for x in imc_dim if x.name == 'D3'),None).size
        M = self.kwargs['accelerator'].cores[0].operational_array.unit.group_depth

        return D1,D2,D3,M


    def extract_network_from_workload(self):
        network = {}
        base_layer = {'K':1, 'C':1, 'FX':1,'FY':1, 'OX':1, 'OY':1, 'Ct':1, 'OXt':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':1}
        i = 0
        for id, layer in enumerate(nx.topological_sort(self.workload)):
            if type(layer) == DummyNode:
                continue  # skip the DummyNodes
            loop_dim_size = layer.layer_attrs['loop_dim_size']
            new_layer = copy.deepcopy(base_layer)
            for k,v in loop_dim_size.items():
                new_layer[k] = v
            new_layer['layer_id'] = layer.id
            network[i] = new_layer

            i += 1

        return network
