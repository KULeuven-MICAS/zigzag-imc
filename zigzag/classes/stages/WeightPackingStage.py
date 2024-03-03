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
from shapely.geometry import Polygon

from zigzag.classes.stages.Stage import Stage
from zigzag.classes.workload.dummy_node import DummyNode
from zigzag.classes.opt.CGA.item import Item, ItemPool
from zigzag.classes.opt.CGA.superitem import *
from zigzag.classes.opt.CGA.macro_bin import MacroBin
from zigzag.classes.opt.CGA.column import *
from zigzag.classes.opt.CGA.weight_rewriting import RewriteAllocation
from zigzag.classes.opt.CGA.utils import plot_item_allocation, prime_factors, vgg_16_network, vgg_16_network_validation

import pickle


logger = logging.getLogger(__name__)

class WeightPackingStage(Stage):

    ## The class constructor
    # Initialization of self.workload.
    def __init__(self, list_of_callables, *, workload, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload


    def weight_tile_allocation(self, ox_unrolling_scheme):
        ox_unrolling_scheme = [(0,16),
                (1,16),
                (2,8),
                (3,4),
                (4,4),
                (5,4),
                (6,2),
                (7,1),
                (8,1),
                (9,1),
                (10,1),
                (11,1),
                (12,1)]


        network = self.extract_network_from_workload()
        network = copy.deepcopy(vgg_16_network_validation)
        solver_status = ""
        D1, D2, D3, M = self.get_IMC_dimension_parameters() 
        kwargs = self.kwargs.copy()
        itempool = ItemPool(D1=D1,D2=D2,D3=D3,M=M,network=network, ox_unrolling_scheme=ox_unrolling_scheme,verbose=0)
        feasible_cfg = itempool.set_init_M(ox_unrolling_scheme)
        if not feasible_cfg:
            logger.error('>>>> Unfeasible settings for D1,D2,D3,M <<<<')
            return [], False, 0, None
        oxu_scheme_str = [f"L{x[0]} OXu {x[1]}" for x in ox_unrolling_scheme] 
        logger.info(f'OX unrolling scheme {oxu_scheme_str}')
        ox_unroll_combs = {}
        item_pool, feasible_tile_configuration, target_layer_index = itempool.generate()
        while not feasible_tile_configuration:
            feasible_tile_configuration, combs = itempool.update_mapping2(target_layer_index)
            ox_unroll_combs[target_layer_index[0]] = combs
            if not feasible_tile_configuration:
                break
            item_pool, feasible_tile_configuration, target_layer_index = itempool.generate()
        if not feasible_tile_configuration:
            logger.error('>>>> Unfeasible settings for D1,D2,D3,M <<<<')
            return [], False, 0, None
        cost_list_iterations = []
        best_cost, cost, weight_writing_cost = float('inf'), float('inf'), float('inf')
        first_iteration = True
        cost_dict = None
        while solver_status not in ['OPTIMAL','FEASIBLE']:
        #    logger.info("===== ItemPool Generation =====")
            si = SuperItemPool(item_pool,verbose=0)
        #    logger.info("===== SuperItemPool Generation =====")
            superitem_pool = si.generate()
        #    logger.info("===== ColumnPool Generation =====")
            column_pool = ColumnPool(D1=D1,D2=D2, network_layers=len(network.keys()),verbose=0)
            column_list = column_pool.generate(superitem_pool)
        #    logger.info("===== Bin Allocation =====")
            macro_bin = MacroBin(height=M, number_of_macros=D3,verbose=0)
            bin_dict, solver_status, not_allocated_item_pool = macro_bin.macro_allocation(column_list)
            #bin_dict, solver_status = macro_bin.pack_macrobin(column_list, fsi, zsl, ol, nki)
            #self.generate_mappings(network, item_pool)
            if solver_status in ['OPTIMAL','FEASIBLE']:
        #        plot_item_allocation(column_list, bin_dict, D3=int(D3), height=int(M), D1=int(D1),D2=int(D2))
                utilization = self.get_utilization(item_pool) / D1 / D2 / D3 / M
                # [EDIT FOR ISSCC VALIDATION]
                cost = self.get_cost(kwargs, {},{})
                cost_list_iterations.append(['Allocated',cost])
                logger.info(f'Allocation cost: {cost:.2e}')
                logger.info(f'>>>> Completed allocation <<<< Utilization {utilization*100:.2f}%')
                break
            else:
                if first_iteration:
                    # Remove not allocated columns from column_list
                    allocated_columns = [j for i in [v for v in bin_dict.values()] for j in i]
                    column_list = [x for x in column_list if x.id in allocated_columns]
                    # Allocate remaining items
                    #ra = RewriteAllocation(bin_dict, column_list, not_allocated_item_pool, network, D1, D2, D3, M)
                    # Estimate cost of rewriting
                    #logger.info(f'Running weight rewriting allocation for {len(not_allocated_item_pool)} items...')
                    # [EDIT FOR ISSCC VALIDATION]
                    #extra_cells, extra_rows = ra.run()
                    #logger.info(f'Extra cells per layer:{extra_cells}')
                    #logger.info(f'Extra rows per layer:{extra_rows}')
#                    weight_writing_cost = self.get_cost(kwargs, extra_cells, extra_rows)
                    weight_writing_cost, cost_dict = self.get_isscc_cost(kwargs, itempool.network, ox_unroll_combs)
                    w_utilization = self.get_utilization(item_pool) / D1 / D2 / D3 / M
                    logger.info(f'Weight rewriting cost: {weight_writing_cost:.3e}, Utilization {w_utilization*100:.2f}%')
                    print(cost_dict)

                    cost_list_iterations.append(['Weight writing',weight_writing_cost])
                    first_iteration = False
                    break
                # Fold M and estimate cost
                feasible_configuration = itempool.update_mapping()
                if not feasible_configuration:
                    logger.error('>>>> Unfeasible settings for D1,D2,D3,M <<<<')
                    break
                item_pool, feasible_tile_configuration, target_layer_index = itempool.generate()
                while not feasible_tile_configuration:
                    feasible_tile_configuration = itempool.update_mapping(target_layer_index)
                    if not feasible_tile_configuration:
                        logger.error('>>>> Unfeasible settings for D1,D2,D3,M <<<<')
                        break
                    item_pool, feasible_tile_configuration, target_layer_index = itempool.generate()
                if not feasible_tile_configuration:
                    logger.error('>>>> Unfeasible settings for D1,D2,D3,M <<<<')
                    return [], False, 0, None

                #self.generate_mappings(network, item_pool)
                #folding_cost = self.get_cost(kwargs, {},{})
                #logger.info(f'Folding cost {folding_cost:.3e}')
                #cost_list_iterations.append(['M folding',folding_cost])
                #if weight_writing_cost < folding_cost:
                #    best_cost = weight_writing_cost

            if not feasible_configuration:
                logger.error('>>>> Unfeasible settings for D1,D2,D3,M <<<<')
                return [], False, 0, None

        if cost < weight_writing_cost:
            best_cost = cost
        else:
            best_cost = weight_writing_cost
            utilization = w_utilization
        return best_cost, True, utilization, cost_dict
        
    

    def get_isscc_cost(self, kwargs, network, dx_combs):
        computation_latency = 0
        weight_loading = 0
        latency = {}
        for ii_n, n in network.items():
            layer_latency = 1
            for k,v in n.items():
                if k[-1] == 't':
                   layer_latency *= v
            latency[ii_n] = [layer_latency]

        total_weights = 0
        for k,v in network.items():
            if k+1 in dx_combs.keys():
                d3_comb = dx_combs[k+1][-1]
                ku = min(np.prod([x[1] for x in d3_comb if x[0] == 'K']), 8)
            else:
                ku = min(v['K'],8)
            latency[k].append(v['K'] * v['C'] * v['FX'] * v['FY'] / 16 / ku)
        
        return sum([sum(v) for v in latency.values()]), latency

    def get_cost(self, kwargs, extra_cells, extra_rows):
        kwargs['workload'] = self.workload
        kwargs['extra_cells'] = extra_cells
        kwargs['extra_rows'] = extra_rows
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        cme_list = []
        for cme, (layer, extra_info) in sub_stage.run():
            cme_list.append(cme)
        cost = sum([x.energy_total * x.latency_total0 for x in cme_list])
        return cost

    
    def get_utilization(self, item_pool):
        return sum([x.volume * x.tile_index for x in item_pool])
    
    def run(self):

        network = self.extract_network_from_workload()
        network = vgg_16_network
        valid_ox_unrolling_scheme = []
        valid_ox_unrolling_scheme_df = []
        cme_list = []
        min_cost = float('inf')

        best_ox_unroll, best_ox_unroll_new = [], []
        for layer_index, layer in network.items():
            ox_pf = [x for x in prime_factors(layer['OXt'])]
            ox_comb = []
            for k in range(1,len(ox_pf)+1):
                for comb in itertools.combinations(ox_pf, k):
                    if np.prod(comb) not in ox_comb:
                        ox_comb.append(np.prod(comb))
            if layer_index == 0:
                ox_comb.insert(0,1)
            ox_comb.sort()
            ox_comb = [(layer_index, x) for x in ox_comb] 
            # [EDIT] ISSCC validation
#            if valid_ox_unrolling_scheme == []:
#                ox_unrolling_scheme_list = [[x] for x in ox_comb]
#            else:
#                ox_unrolling_scheme_list = itertools.product(*[ox_comb, valid_ox_unrolling_scheme])
#                ox_unrolling_scheme_list = [[x[0]] + x[1] for x in ox_unrolling_scheme_list]
#            for ox_unrolling_scheme in ox_unrolling_scheme_list:
            best_ox_unroll = best_ox_unroll_new
            for ox_c in ox_comb:
                if ox_c[1] == 112:
                    continue
                best_ox_unroll_cp = copy.deepcopy(best_ox_unroll)
                best_ox_unroll_cp.append(ox_c)
                # [EDIT] ISSCC validation
                #cme, feasible, utilization, cost_dict = self.weight_tile_allocation(ox_unrolling_scheme)
                cme, feasible, utilization, cost_dict = self.weight_tile_allocation(best_ox_unroll_cp)

                # [EDIT FOR ISSCC VALIDATION]
                if not feasible:
                    break
                #valid_ox_unrolling_scheme_df.append({'cost':cme, 'OXu':ox_unrolling_scheme, 'utilization':cost_dict})
                valid_ox_unrolling_scheme_df.append({'cost':cme, 'OXu':best_ox_unroll_cp, 'utilization':cost_dict})
                #valid_ox_unrolling_scheme.append(ox_unrolling_scheme)
                valid_ox_unrolling_scheme.append(best_ox_unroll_cp)

                # Heuristic introduced to avoid evaluating suboptimal OX unrollings
                if cme < min_cost:
                    best_ox_unroll_new = copy.deepcopy(best_ox_unroll_cp)
                    min_cost = cme
                    cme_list.append((best_ox_unroll_cp,cme,cost_dict))

        cme_list.sort(key=lambda x: x[1])
        breakpoint()
        with open('isscc_validation.pkl','wb') as infile:
            pickle.dump(cme_list, infile)
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
                if k in ['OX','OY']:
                    new_layer[f'{k}t'] = v
                else:
                    new_layer[k] = v
            new_layer['layer_id'] = layer.id
            network[i] = new_layer

            i += 1

        return network


if __name__ == "__main__":
    from shapely.geometry import Polygon

    p = Polygon([(1,1),(2,2),(4,2),(3,1)])
    q = Polygon([(1.5,2),(3,5),(5,4),(3.5,1)])
    breakpoint()
    print(p.intersects(q))  # True
    print(p.intersection(q).area)  # 1.0
