from typing import Generator, Callable, List, Tuple, Any
from classes.stages.Stage import Stage
import copy
import numpy as np
import logging
import networkx as nx
from classes.cost_model.cost_model import CostModelEvaluation
from classes.hardware.architecture.accelerator import Accelerator
from classes.mapping.spatial.spatial_mapping import SpatialMapping
from classes.mapping.temporal.temporal_mapping import TemporalMapping
from classes.workload.layer_node import LayerNode
import classes.io.input_config as inputs
logger = logging.getLogger(__name__)
import pdb


def merge_loops(temporal_mapping):
    tm_merged = {}
    tm = temporal_mapping.mapping_dic_origin
    for op in tm.keys():
        tm_merged[op] = []
        for lev in tm[op]:
            tm_level = []
            for ii_tm, tmx in enumerate(lev):
                if ii_tm > 0:
                    if tmx[0] == lev[ii_tm-1][0]:
                        tm_level[-1][1] *= tmx[1]
                    else:
                        tm_level.append([tmx[0], tmx[1]])
                else:
                    tm_level.append([tmx[0], tmx[1]])
            tm_merged[op].append(tm_level)
    for op in tm_merged.keys():
        tm_merged[op] = [[tuple(x) for x in lev] for lev in tm_merged[op]] 


    temporal_mapping.mapping_dic_origin = tm_merged
    return tm_merged


class CostModelStage(Stage):
    """
    Pipeline stage that calls a cost model to evaluate a mapping on a HW config.
    """
    def __init__(self, list_of_callables:List[Callable], *, accelerator, layer, spatial_mapping, temporal_mapping, **kwargs):
        """
        Initializes the cost model stage given main inputs
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator, self.layer, self.spatial_mapping, self.temporal_mapping =\
            accelerator, layer, spatial_mapping, temporal_mapping

    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the cost model stage by calling the internal zigzag cost model with the correct inputs.
        """
        accelerator_copy = copy.deepcopy(self.accelerator)
        tm = copy.deepcopy(self.temporal_mapping.mapping_dic_origin)
        sm = copy.deepcopy(self.spatial_mapping.mapping_dict_origin)
        mem_levels_to_be_removed = []
        ''' remove higher mem_level when size of lower mem_level is enough to hold all operands '''
        for operand in ['I','O','A']:
            #if operand in ['B', 'W']:
            #    if accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS']<16:
            #        continue
            #    try:
            #        ''' mobilenet '''
            #        if accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 16:
            #            if self.layer.id not in [28]:
            #                continue
            #        elif accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 32:
            #            if self.layer.id not in [28, 32, 36]:
            #                continue
            #        elif accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 64:
            #            if self.layer.id not in [28, 32, 36, 40, 44, 48]:
            #                continue
            #        elif accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 128:
            #            if self.layer.id not in [16, 20, 24, 28, 32, 36, 40, 44, 48, 52]:
            #                continue
            #        ''' ae '''
                    #if accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 16:
                    #    if self.layer.id not in [2,8]:
                    #        accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] -=9
                    #        continue
                    #elif accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 32:
                    #    if self.layer.id not in [2,4,6,8]:
                    #        accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] -=25
                    #        continue
                    #elif accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 64:
                    #    if self.layer.id not in [2,4,6,8,10,12,14,16]:
                    #        accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] -=57
                    #        continue
                    #elif accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 128:
                    #    if self.layer.id not in [0,2,4,6,8,10,12,14,16]:
                    #        continue
                    #''' ds-cnn '''
                    #if self.layer.loop_dim_size['G'] > 1:
                    #    if accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS']>25:
                    #        accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] -=25
                    #    continue # do not remove dram level
                #except: # fc layer
                #    pass
            try:
                md_op = self.layer.memory_operand_links[operand]
                for ii_l, l in enumerate(self.temporal_mapping.mapping_dic_origin[operand]):
                    if not any(self.temporal_mapping.mapping_dic_origin[operand][ii_l:]):
                        mem_level = accelerator_copy.cores[0].mem_hierarchy_dict[md_op][ii_l].name
                        mem_levels_to_be_removed.append((md_op, mem_level, ii_l))
            except KeyError:
                continue
        mem_levels_to_be_removed = set(mem_levels_to_be_removed)
        for mem_level in mem_levels_to_be_removed:
            ml = next((x for x in accelerator_copy.cores[0].memory_hierarchy.nodes if x.name == mem_level[1]))
            ml_operands = ml.operands
            idx_op = ml_operands.index(mem_level[0])
            ml_level_of_operands = ml.mem_level_of_operands
            ml_port_alloc_raw = list(ml.port_alloc_raw)
            del ml_port_alloc_raw[idx_op]
            ml_port_alloc_raw = tuple(ml_port_alloc_raw)
            ml_operands.remove(mem_level[0])
            del ml_level_of_operands[mem_level[0]]

            mlx = copy.deepcopy(ml)
            mlx.operands = ml_operands
            mlx.mem_level_of_operands = ml_level_of_operands
            mlx.port_alloc_raw = ml_port_alloc_raw
            nx.set_node_attributes(accelerator_copy.cores[0].memory_hierarchy, {ml:{'operands':ml_operands, 'mem_level_of_operands':ml_level_of_operands, 'port_alloc_raw':ml_port_alloc_raw}})
            ml_index = accelerator_copy.cores[0].memory_hierarchy.mem_instance_list.index(ml)
            if len(ml_level_of_operands) != 0:
                accelerator_copy.cores[0].memory_hierarchy.mem_instance_list[ml_index] = mlx
            else: #completely remove this mem instance
                accelerator_copy.cores[0].memory_hierarchy.remove_node(ml)
                del accelerator_copy.cores[0].memory_hierarchy.mem_instance_list[ml_index]


        #ml_tmp = [x for x in accelerator_copy.cores[0].memory_hierarchy.nodes if 'I1' in x.operands]
        #ml_dram = ml_tmp[-1]
        #ml_dram_index = accelerator_copy.cores[0].memory_hierarchy.mem_instance_list.index(ml_dram)
        #mlx = copy.deepcopy(ml_dram)
        #tmm = [j for i in sm['O'] for j in i]

        #if 'W' in self.layer.operand_precision.keys():
        #    dram_bw = np.prod([x[1] for x in tmm if x[0] in ['K']])* self.layer.operand_precision['W']
        #else:
        #    dram_bw = np.prod([x[1] for x in tmm if x[0] in ['K']])* self.layer.operand_precision['B']
        #mlx.port_list[0].port_bw = dram_bw
        #mlx.port_list[0].port_bw_min = dram_bw
        #accelerator_copy.cores[0].memory_hierarchy.mem_instance_list[ml_dram_index] = mlx

        accelerator_copy.cores[0].generate_memory_hierarchy_dict()
        accelerator_copy.cores[0].generate_memory_sharing_list()

        #accelerator_copy.cores[0].mem_r_bw_dict['I1'][-1] = dram_bw
        #accelerator_copy.cores[0].mem_r_bw_min_dict['I1'][-1] = dram_bw

        for operand in ['I','O','A']:
            #if operand in ['B', 'W']:
            #    if accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS']<16:
            #        continue
            #    try:
            #        ''' mobilenet '''
            #        if accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 16:
            #            if self.layer.id not in [28]:
            #                continue
            #        elif accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 32:
            #            if self.layer.id not in [28, 32, 36]:
            #                continue
            #        elif accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 64:
            #            if self.layer.id not in [28, 32, 36, 40, 44, 48]:
            #                continue
            #        elif accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 128:
            #            if self.layer.id not in [16, 20, 24, 28, 32, 36, 40, 44, 48, 52]:
            #                continue
                    #''' ae '''
                    #if accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 16:
                    #    if self.layer.id not in [2,8]:
                    #        accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] -=9
                    #        continue
                    #elif accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 32:
                    #    if self.layer.id not in [2,4,6,8]:
                    #        accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] -=25
                    #        continue
                    #elif accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 64:
                    #    if self.layer.id not in [2,4,6,8,10,12,14,16]:
                    #        accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] -=57
                    #        continue
                    #elif accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 128:
                    #    if self.layer.id not in [0,2,4,6,8,10,12,14,16]:
                    #        continue
                    #''' resnet8 '''
                    #if accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] == 32:
                    #    if self.layer.id != 15:
                    #        continue
                    #''' ds-cnn '''
                    #if self.layer.loop_dim_size['G'] > 1:
                    #    if accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS']>25:
                    #        accelerator_copy.cores[0].memory_hierarchy.operational_array.unit.cost['CORE_ROWS'] -=25
                    #    continue # do not remove dram level
                #except:
                #    pass
            try:
                for ii_l, l in enumerate(tm[operand]):
                    if not any(tm[operand][ii_l:]):
                        tm[operand] = tm[operand][:ii_l]
                        sm[operand] = sm[operand][:ii_l+1]
            except KeyError:
                continue
        sm_new = SpatialMapping(sm, self.layer)
        tm_new = TemporalMapping(tm, self.layer)
        self.cme = CostModelEvaluation(accelerator=accelerator_copy,
                                       layer=self.layer,
                                       spatial_mapping=sm_new,
                                       temporal_mapping=merge_loops(tm_new)
                                       )
        yield (self.cme, None)

    def is_leaf(self) -> bool:
        return True
