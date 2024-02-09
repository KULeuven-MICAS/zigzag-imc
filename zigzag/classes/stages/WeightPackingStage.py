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
from zigzag.classes.opt.CGA.utils import plot_item_allocation



logger = logging.getLogger(__name__)

class WeightPackingStage(Stage):

    ## The class constructor
    # Initialization of self.workload.
    def __init__(self, list_of_callables, *, workload, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload

    def run(self):

        # LAYERS MUST START FROM INDEX ZERO!
        network = self.extract_network_from_workload()
        solver_status = ""
        D1, D2, D3, M = self.get_IMC_dimension_parameters() 

        itempool = ItemPool(D1=D1,D2=D2,D3=D3,M=M,network=network)
        itempool.set_init_M()
        while solver_status not in ['OPTIMAL','FEASIBLE']:
            logger.info("===== ItemPool Generation =====")
            item_pool, feasible_tile_configuration, target_layer_index = itempool.generate()
            while not feasible_tile_configuration:
                itempool.update_network(target_layer_index)
                item_pool, feasible_tile_configuration, target_layer_index = itempool.generate()
            si = SuperItemPool(item_pool)
            logger.info("===== ItemPool Generation Done =====")
            logger.info("===== SuperItemPool Generation =====")
            superitem_pool = si.generate()
            logger.info("===== LayerPool Generation =====")
            layer_pool = LayerPool(D1=D1,D2=D2, network_layers=len(network.keys()))
            layer_list = layer_pool.generate(superitem_pool)
#            fsi, zsl, ol, nki = layer_pool.generate_cp_parameters()
            logger.info("===== Bin Allocation =====")
            macro_bin = MacroBin(height=M, number_of_macros=D3)
            bin_dict, solver_status = macro_bin.macro_allocation(layer_list)

            #bin_dict, solver_status = macro_bin.pack_macrobin(layer_list, fsi, zsl, ol, nki)
            if solver_status in ['OPTIMAL','FEASIBLE']:
                plot_item_allocation(layer_list, bin_dict, D3=int(D3), height=int(M), D1=int(D1),D2=int(D2))
                logger.success('>>>> Completed allocation <<<<')
                break
            else:
                feasible_configuration = itempool.update_network()
            if not feasible_configuration:
                logger.error('>>>> Unfeasible settings for D1,D2,D3,M <<<<')
                exit()

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.workload, **kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, (layer, extra_info)

    def get_IMC_dimension_parameters(self):
        imc_dim = self.kwargs['accelerator'].cores[0].operational_array.dimensions
        D1 = next((x for x in imc_dim if x.name == 'D1'),None).size
        D2 = next((x for x in imc_dim if x.name == 'D2'),None).size
        D3 = next((x for x in imc_dim if x.name == 'D3'),None).size
        M = self.kwargs['accelerator'].cores[0].operational_array.unit.group_depth

        return D1,D2,D3,M


    def extract_network_from_workload(self):
        network = {}
        base_layer = {'K':1, 'C':1, 'FX':1,'FY':1, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1}
        i = 0
        for id, layer in enumerate(nx.topological_sort(self.workload)):
            if type(layer) == DummyNode:
                continue  # skip the DummyNodes
            loop_dim_size = layer.layer_attrs['loop_dim_size']
            new_layer = copy.deepcopy(base_layer)
            for k,v in loop_dim_size.items():
                new_layer[k] = v
            network[i] = new_layer
            i += 1

        return network
