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
from zigzag.classes.opt.CGA.utils import plot_item_allocation, prime_factors



logger = logging.getLogger(__name__)

class WeightPackingStage(Stage):

    ## The class constructor
    # Initialization of self.workload.
    def __init__(self, list_of_callables, *, workload, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload

    def weight_reloading_cells(self, column_list, not_allocated_columns, bin_dict, D2, M, network, bin_dict_new_alloc):
        bin_dict_copy = bin_dict.copy()
        extra_cells_per_layer = {}
        extra_rows_per_layer = {}
        for n in network.keys():
            extra_cells_per_layer[n] = 0
            extra_rows_per_layer[n] = 0
        for nal in not_allocated_columns:
            nal_layer_index = set()
            nal.sort(key=lambda x:x.depth)
            base_z = M - sum([x.height for x in nal])
            item_list = []
            # Create item coordinates
            for column in nal:
                nal_layer_index.update(column.layer_index_set)
                for superitem in column.superitem_set:
                    base_z_si = base_z
                    for item in superitem.item_set:
                        # Check if item overlaps with allocated items
                        item.x_pos = superitem.x_pos
                        item.y_pos = D2 - superitem.depth
                        item.z_pos = base_z_si
                        item_list.append(item)
                        base_z_si += item.height
                base_z += column.height

            # Create set of network layer indices of bin
            # b is a list of columns id
            bin_allocation_index = next((k for k,v in bin_dict_new_alloc.items() if set([x[0] for x in v]) == set([x.id for x in nal])),None)
            bin_columns = [next((x for x in column_list if x.id == lx),None) for lx in bin_dict[bin_allocation_index]]
            bin_layers_indices = set()
            for bx in bin_columns:
                bin_layers_indices.update(bx.layer_index_set)
            for cx in nal:
                bin_dict_copy[bin_allocation_index].append(cx.id)
            item_list_bin = []
            base_z = 0
            for column in bin_columns:
                for superitem in column.superitem_set:
                    base_z_si = base_z
                    for item in superitem.item_set:
                        item.x_pos = superitem.x_pos
                        item.y_pos = superitem.y_pos
                        item.z_pos = base_z_si
                        item_list_bin.append(item)
                        base_z_si += item.height
                base_z += column.height
            for item_nal in item_list:
                for item_al in item_list_bin:
                    # compute overlap
                    overlap_volume = max(min(item_al.x_pos + item_al.width, item_nal.x_pos + item_nal.width) - \
                        max(item_al.x_pos, item_nal.x_pos), 0)
                    overlap_volume *= max(min(item_al.y_pos + item_al.depth, item_nal.y_pos + item_nal.depth) - \
                        max(item_al.y_pos, item_nal.y_pos), 0)
                    overlap_volume *= max(min(item_al.z_pos + item_al.height, item_nal.z_pos + item_nal.height) - \
                        max(item_al.z_pos, item_nal.z_pos), 0)

                    overlap_area = max(min(item_al.y_pos + item_al.depth, item_nal.y_pos + item_nal.depth) - \
                        max(item_al.y_pos, item_nal.y_pos), 0)
                    overlap_area *= max(min(item_al.z_pos + item_al.height, item_nal.z_pos + item_nal.height) - \
                        max(item_al.z_pos, item_nal.z_pos), 0)
                    if overlap_volume > 0:
                        extra_cells_per_layer[item_nal.layer_index] += overlap_volume
                        extra_cells_per_layer[item_al.layer_index] += overlap_volume
                        extra_rows_per_layer[item_nal.layer_index] += overlap_area
                        extra_rows_per_layer[item_al.layer_index] += overlap_area

        print(extra_cells_per_layer)
        return bin_dict_copy, extra_cells_per_layer, extra_rows_per_layer


    def imc_weight_reloading_rows(self, column_list, not_allocated_columns, bin_dict, D2, M):
        
        bin_dict_copy = copy.deepcopy(bin_dict)
        bin_dict_new_alloc = {}
        for k,b in bin_dict.items():
            bin_columns = [next((x for x in column_list if x.id == lx),None) for lx in b]
            bin_dict_new_alloc[k] = []
        extra_rows_per_layers = {}
        # each element in not_allocated_columns is a list of columns
        for nal in not_allocated_columns:

            # Create polygon of not allocated columns
            # Create set of network layer indices
            min_intersection_area = float('inf')
            nal.sort(key=lambda x:x.depth)
            nal_layer_index = set()
            nal_polygon_vertices = []
            bottom_x = M - sum([x.height for x in nal])
            bottom_y = D2 
            nal_polygon_vertices.append((M, D2))
            nal_polygon_vertices.append((bottom_x, D2))
            for ii_nl, nl in enumerate(nal):
                bottom_y = D2 - nl.actual_depth
                bottom_x = M - sum([x.height for x in nal[ii_nl:]])
                nal_polygon_vertices.append((bottom_x, bottom_y))
                bottom_x = M - sum([x.height for x in nal[ii_nl+1:]])
                nal_polygon_vertices.append((bottom_x, bottom_y))
                nal_layer_index.update(nl.layer_index_set)
            not_allocated_polygon = Polygon(nal_polygon_vertices)
            for k,b in bin_dict_copy.items():
                # Create set of network layer indices of bin
                bin_columns = [next((x for x in column_list if x.id == lx),None) for lx in b]
                bin_layers_indices = set()
                for bx in bin_columns:
                    bin_layers_indices.update(bx.layer_index_set)
                if bin_layers_indices.intersection(nal_layer_index) == set():
                    # Create polygon of layers allocated in bin
                    b_polygon_vertices = []
                    top_x = sum([x.height for x in bin_columns])
                    top_y = 0
                    b_polygon_vertices.append((top_x, top_y))
                    b_polygon_vertices.append((0,0))
                    for ii_bx, bx in enumerate(bin_columns):
                        top_y = bx.actual_depth
                        top_x = sum([x.height for x in bin_columns[:ii_bx]])
                        b_polygon_vertices.append((top_x, top_y))
                        top_x = sum([x.height for x in bin_columns[:ii_bx+1]])
                        b_polygon_vertices.append((top_x, top_y))
                    allocated_polygon = Polygon(b_polygon_vertices)
                    intersection_area = allocated_polygon.intersection(not_allocated_polygon).area
                    if intersection_area < min_intersection_area:
                        min_intersection_area = intersection_area
                        best_bin = k
            if min_intersection_area == float('inf'):
                breakpoint()
            for nl in nal:
                bin_dict_new_alloc[best_bin].append((nl.id, min_intersection_area, nl.layer_index_set))
                bin_dict_copy[best_bin].append(nl.id)

            # Assign extra rows to be written to one of the network layers that belong to the nal
            extra_rows_per_layers[list(nal_layer_index)[0]] = min_intersection_area
        print('Extra rows per layer', extra_rows_per_layers)
        return bin_dict_new_alloc, extra_rows_per_layers

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
        #    logger.info("===== ColumnPool Generation =====")
            column_pool = ColumnPool(D1=D1,D2=D2, network_layers=len(network.keys()))
            column_list = column_pool.generate(superitem_pool)
        #    logger.info("===== Bin Allocation =====")
            macro_bin = MacroBin(height=M, number_of_macros=D3)
            bin_dict, solver_status, not_allocated_columns = macro_bin.macro_allocation(column_list)
            #bin_dict, solver_status = macro_bin.pack_macrobin(column_list, fsi, zsl, ol, nki)
            if solver_status in ['OPTIMAL','FEASIBLE']:
        #        plot_item_allocation(column_list, bin_dict, D3=int(D3), height=int(M), D1=int(D1),D2=int(D2))
                self.generate_mappings(network, item_pool)
                utilization = self.get_utilization(item_pool) / D1 / D2 / D3 / M
                logger.info(f'>>>> Completed allocation <<<< Utilization {utilization*100:.2f}%')
                break
            else:
                # Estimate cost of weight reloading
                bin_dict_with_rewriting, extra_write_row_per_network_layer = self.imc_weight_reloading_rows(column_list, not_allocated_columns, bin_dict, D2, M)
                bin_dict_with_rewriting, extra_write_cells_per_network_layer, extra_rows = self.weight_reloading_cells(column_list, not_allocated_columns, bin_dict, D2, M, network,  bin_dict_with_rewriting)

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


if __name__ == "__main__":
    from shapely.geometry import Polygon

    p = Polygon([(1,1),(2,2),(4,2),(3,1)])
    q = Polygon([(1.5,2),(3,5),(5,4),(3.5,1)])
    print(p.intersects(q))  # True
    print(p.intersection(q).area)  # 1.0
