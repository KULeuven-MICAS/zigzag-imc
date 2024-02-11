import numpy as np
from ortools.sat.python import cp_model
from loguru import logger
import copy
from tqdm import tqdm


class Bin():
    def __init__(self):
        self.height = 0
        self.layer_set = set()
        self.layer_index_set = set()
        self.density = 0
        self.volume = 0
        self.id = 0

    def add_layer(self, layer):
        self.layer_set.add(layer)
        self.height += layer.height
        self.volume += layer.volume
        self.layer_index_set.update(layer.layer_index_set)


class LayerLite():
    def __init__(self, height, volume, layer_index_set, id):
        self.height = height
        self.volume = volume
        self.layer_index_set = layer_index_set
        self.id = id

    def __eq__(self, other):
        if self.height == other.height and self.volume == other.volume and self.id == other.id and self.layer_index_set == other.layer_index_set:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.id)


class MacroBin():
    def __init__(self, height, number_of_macros):
        self.height = height
        self.D3 = number_of_macros


    @staticmethod
    def macro_allocation_recursive(bin_alloc, max_height, layer_list, bin_alloc_list):
        if layer_list == []:
            bin_alloc_layer_index_set = [[x.layer_index_set for x in b.layer_set] for b in bin_alloc_list]
            bin_layer_index_set = [x.layer_index_set for x in bin_alloc.layer_set]
            for b in bin_alloc_layer_index_set:
                if all([x in b for x in bin_layer_index_set]) and all([x in bin_layer_index_set for x in b]):
                    return
            bin_alloc_list.append(bin_alloc)
        else:
            for layer in layer_list:
    #            if layer.layer_index_set.intersection(bin_alloc.layer_index_set) != set():
    #                MacroBin.macro_allocation_recursive(bin_alloc, max_height, [], bin_alloc_list)
                if layer.height + bin_alloc.height > max_height:
                    MacroBin.macro_allocation_recursive(bin_alloc, max_height, [], bin_alloc_list)
                else:
                    bin_alloc_copy = copy.deepcopy(bin_alloc)
                    bin_alloc_copy.add_layer(layer)
                    
                    # Check if superset combination already exists
                    for b in bin_alloc_list:
                        if all([x in b.layer_index_set for x in bin_alloc_copy.layer_index_set]):
                            return
                        #MacroBin.macro_allocation_recursive(bin_alloc_copy, max_height, [], bin_alloc_list)
                    #layer_list_copy = [x for x in layer_list if x != layer_copy]
                    layer_list_copy = [x for x in layer_list if x != layer]
                    MacroBin.macro_allocation_recursive(bin_alloc_copy, max_height, layer_list_copy, bin_alloc_list)


    def macro_allocation(self, layer_list):
        total_bin_alloc_list = []
        layer_alloc_list = []
        layer_lite_list = []
        for l in layer_list:
            layer_lite_list.append(LayerLite(l.height, l.get_volume(), l.layer_index_set, l.id))
        
        # progress bar
        init_range = len(layer_lite_list)
#        pbar = tqdm(total=init_range)

        while layer_lite_list != []:
            valid_layer_subsets = [[layer_lite_list[0]]]
            for l in layer_lite_list:
                assigned = False
                for subset in valid_layer_subsets:
                    if all([l.layer_index_set.intersection(x.layer_index_set) == set() for x in subset]):
                        assigned = True
                        subset.append(l)
                        break
                if not assigned:
                    valid_layer_subsets.append([l])
            valid_layer_subsets.sort(key=lambda x : len(x),reverse=True)
            bb = False
            for layer_lite_list_subset in valid_layer_subsets:
                layer_lite_list_subset.sort(key=lambda x : x.height)
                l = layer_lite_list_subset[0]
                bin_alloc_list = []
                bin_alloc = Bin()
                bin_alloc.add_layer(l)
                layer_list_copy = copy.deepcopy(layer_lite_list_subset)
                layer_list_copy = [x for x in layer_list_copy if x != l]
                MacroBin.macro_allocation_recursive(bin_alloc, self.height, layer_list_copy, bin_alloc_list)
                if bin_alloc_list != []:
                    bin_alloc_list.sort(key=lambda x : x.volume, reverse=True)
                    best_b = bin_alloc_list[0]
                    break

            best_b_layer_index = [x.layer_index_set for x in best_b.layer_set]
            newly_added_layers = []
            for subset in valid_layer_subsets:
                subset_layer_index = [x.layer_index_set for x in subset]
                if all([x in subset_layer_index for x in best_b_layer_index]):# and all([x in best_b_layer_index for x in subset_layer_index]):
                    total_bin_alloc_list.append([x.id for x in subset])
                    newly_added_layers.append([next((xx for xx in layer_list if xx.id == x.id),None) for x in subset])
                    layer_alloc_list += [x.id for x in subset]
                    layer_lite_list = [x for x in layer_lite_list if x.id not in layer_alloc_list]
            if len(total_bin_alloc_list) > self.D3:
#                pbar.close()
#                logger.warning(f'Exceeded number of D3')
                return None, "UNFEASIBLE" 
            #pbar.update(init_range - len(layer_lite_list))
            init_range = len(layer_lite_list)
        #pbar.close()
        bin_dict =  {}
        for i in range(self.D3):
            bin_dict[i] = []
            if len(total_bin_alloc_list) > i:
                for ii_l, l in enumerate(total_bin_alloc_list[i]):
                    bin_dict[i].append(l)

        #logger.info(f'Generated Bins #{len(total_bin_alloc_list)}')#; Layers to be assigned: {len(layer_lite_list)}')
        return bin_dict, "OPTIMAL"


    def pack_macrobin(self, layer_list, fsi, zsl, ol, nki):
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        n_superitems, n_items = fsi.shape
        n_network_layers, _ = nki.shape

        # Variable definition
        tb, al, ulb = dict(), dict(), dict()
        for b in range(self.D3):
            tb[b] = model.NewBoolVar(f"t_{b}")
        for l in range(len(layer_list)):
            #al[l] = model.NewBoolVar(f'a_{l}')
            for b in range(self.D3):
                ulb[l,b] = model.NewBoolVar(f'u_{l}_{b}')
        
        # Constraints
        # Ensure that every item is included in exactly one layer
        for i in range(n_items):
            model.Add(
                cp_model.LinearExpr.Sum(
                    fsi[s, i] * zsl[s, l] * ulb[l, b] for s in range(n_superitems) for l in range(len(layer_list)) for b in range(self.D3)
                )
                == 1
            )

        for k in range(n_network_layers):
            for b in range(self.D3):
                model.Add(
                    cp_model.LinearExpr.Sum(
                        nki[k, i] * fsi[s, i] * zsl[s, l] * ulb[l, b] for i in range(n_items) for s in range(n_superitems) for l in range(len(layer_list))
                    )
                    <= 1
                )


        # Ensure that height of layer combination does not exceed bin height
        for b in range(self.D3):
            model.Add(
                cp_model.LinearExpr.Sum(
                    ol[l] * ulb[l, b] for l in range(len(layer_list))
                )
                <= self.height
            )
        for l in range(len(layer_list)):
            for b in range(self.D3):
                model.Add(ulb[l, b] <= tb[b])
                #model.Add(ulb[l, b] <= al[l])

        obj = cp_model.LinearExpr.Sum(tb[b] for b in range(self.D3))
        model.Minimize(obj)

        # Set solver parameters
        num_workers = 4
        solver.parameters.num_search_workers = num_workers
        solver.parameters.log_search_progress = True
        #solver.parameters.search_branching = cp_model.FIXED_SEARCH

        # Solve
        status = solver.Solve(model)
        bin_dict = {} 

        if solver.StatusName(status) == 'OPTIMAL' or solver.StatusName(status) == 'FEASIBLE':
            for b in range(self.D3):
                bin_dict[b] = []
                for l in range(len(layer_list)):
                    if solver.Value(ulb[l,b]) == 1:
                        bin_dict[b].append(l)
        return bin_dict, solver.StatusName(status)

        

