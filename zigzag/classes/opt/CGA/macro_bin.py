import numpy as np
from ortools.sat.python import cp_model
from loguru import logger
import copy


class Bin():
    def __init__(self):
        self.height = 0
        self.column_set = set()
        self.layer_index_set = set()
        self.density = 0
        self.volume = 0
        self.id = 0

    def add_column(self, column):
        self.column_set.add(column)
        self.height += column.height
        self.volume += column.volume
        self.layer_index_set.update(column.layer_index_set)


class ColumnLite():
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
    def __init__(self, height, number_of_macros, verbose):
        self.height = height
        self.D3 = number_of_macros
        self.verbose = verbose


    @staticmethod
    def macro_allocation_recursive(bin_alloc, max_height, column_list, bin_alloc_list):
        if column_list == []:
            # Check again
            layer_index_set_total = [[x.layer_index_set for x in b.column_set] for b in bin_alloc_list]
            layer_index_set = [x.layer_index_set for x in bin_alloc.column_set]
            for b in layer_index_set_total:
                if all([x in b for x in layer_index_set]) and all([x in layer_index_set for x in b]):
                    return
            bin_alloc_list.append(bin_alloc)
        else:
            for column in column_list:
                if column.layer_index_set.intersection(bin_alloc.layer_index_set) != set():
                    MacroBin.macro_allocation_recursive(bin_alloc, max_height, [], bin_alloc_list)
                elif column.height + bin_alloc.height > max_height:
                    MacroBin.macro_allocation_recursive(bin_alloc, max_height, [], bin_alloc_list)
                else:
                    bin_alloc_copy = copy.deepcopy(bin_alloc)
                    bin_alloc_copy.add_column(column)
                    
                    # Check if superset combination already exists
                    for b in bin_alloc_list:
                        if all([x in b.layer_index_set for x in bin_alloc_copy.layer_index_set]):
                            return
                        #MacroBin.macro_allocation_recursive(bin_alloc_copy, max_height, [], bin_alloc_list)
                    #column_list_copy = [x for x in column_list if x != column_copy]
                    column_list_copy = [x for x in column_list if x != column]
                    MacroBin.macro_allocation_recursive(bin_alloc_copy, max_height, column_list_copy, bin_alloc_list)


    def macro_allocation(self, column_list):
        total_bin_alloc_list = []
        column_alloc_list = []
        column_lite_list = []
        for l in column_list:
            column_lite_list.append(ColumnLite(l.height, l.get_volume(), l.layer_index_set, l.id))
        
        # progress bar
        init_range = len(column_lite_list)
#        pbar = tqdm(total=init_range)

        while column_lite_list != []:
            valid_column_subsets = [[column_lite_list[0]]]
            # Create sets of columns where there are not more than one item per network column
            for l in column_lite_list[1:]:
                assigned = False
                for subset in valid_column_subsets:
                    if all([l.layer_index_set.intersection(x.layer_index_set) == set() for x in subset]):
                        assigned = True
                        subset.append(l)
                        break
                if not assigned:
                    valid_column_subsets.append([l])

            # Sort the sets of columns by the number of columns they contain
            # Evaluate the largest sets of columns first; this is done so that if they are found to be fit,
            # larger number of columns are removed from the list of columns to be allocated, speeding up the search
            valid_column_subsets.sort(key=lambda x : len(x),reverse=True)
            for column_lite_list_subset in valid_column_subsets:
                # Iterate over valid column allocations
                # For each set of valid column allocation, find best combination that fits in the bin height
                # and that maximizes volume
                column_lite_list_subset.sort(key=lambda x : x.height)
                # Start by allocating the "least tall" column; this is the minimum allocation possible
                # Any other combination is a taller set of columns
                l = column_lite_list_subset[0]
                bin_alloc_list = []
                bin_alloc = Bin()
                bin_alloc.add_column(l)
                column_list_copy = copy.deepcopy(column_lite_list_subset)
                column_list_copy = [x for x in column_list_copy if x != l]
                # Recursively look for combinations of column that fit in the bin height
                MacroBin.macro_allocation_recursive(bin_alloc, self.height, column_list_copy, bin_alloc_list)
                # If the recursive allocation has found at least one valid column allocation
                # Adopt the most voluminous valid allocation
                if bin_alloc_list != []:
                    bin_alloc_list.sort(key=lambda x : x.volume, reverse=True)
                    best_b = bin_alloc_list[0]
                    break

            # Find all other valid column allocations that have the same set of column indices
            # This just means that those columns allocations also are the most voluminous ones
            # and with identical configuration, just with different tile index per item

            # best_b_column_index is a list of layer_index_set of the best column allocation found
            # Very ugly name, sorry
            # e.g. [{1}, {2}, {5}]
            best_b_column_index = [x.layer_index_set for x in best_b.column_set]

            new_total_bin_alloc_list, new_column_alloc_list = [], []
            for subset in valid_column_subsets:
                subset_column_index = [x.layer_index_set for x in subset]

                # Check if the best column allocation is at most a subset of the selected valid column allocation
                if all([x in subset_column_index for x in best_b_column_index]):# and all([x in best_b_column_index for x in subset_column_index]):

                    # Select for allocation only the subset of columns that matches with the best allocation found
                    new_total_bin_alloc_list.append([x.id for x in subset if x.layer_index_set in best_b_column_index])
                    new_column_alloc_list += [x.id for x in subset if x.layer_index_set in best_b_column_index]

            # If the number of bins allocated up until this point exceed the available ones, break the loop
            # return the allocated columns until that point and the not allocated ones
            if len(total_bin_alloc_list) + len(new_total_bin_alloc_list) > self.D3:
                # Update total bin alloc list with those bins that still fit in D3
                valid_column_subsets = [[next((x for x in column_list if x.id == lx),None) for lx in y] for y in total_bin_alloc_list]
                init_len_bin_alloc = len(total_bin_alloc_list)
                for i in range(self.D3 - len(total_bin_alloc_list)):
                    valid_column_subsets.append([])
                new_total_bin_alloc = {k:[] for k in range(self.D3)}
                # Create sets of columns where there are not more than one item per network column
                # Theoretically it should be done recursively here as well
                # Column lite list is a list of columns still to be allocated
                for l in column_lite_list:
                    assigned = False
                    # Get column corresponding to the column lite
                    cl = next((x for x in column_list if x.id == l.id),None)
                    # Iterate for the allocated bins
                    for ii_bin, subset in enumerate(valid_column_subsets):
                        # If the intersection between the layer indices of the allocated bin with the selected column is empty:
                        if all([l.layer_index_set.intersection(x.layer_index_set) == set() for x in valid_column_subsets[self.D3 - ii_bin - 1]]):
                            # the sum of previously allocated columns + selected column is less than height:
                            if sum([x.height for x in new_total_bin_alloc[self.D3 - ii_bin - 1]]) + l.height <= self.height:
                                assigned = True
                                new_total_bin_alloc[self.D3 - ii_bin - 1].append(cl)
                                break
                    if not assigned:
                        new_total_bin_alloc[self.D3 - ii_bin - 1].append(cl)

                #print('Allocation', total_bin_alloc_list)
                #print('Still to be allocated', [[x.id for x in y] for y in new_total_bin_alloc.values()])
                #print('Still to be allocated', [[x.layer_index_set for x in y] for y in new_total_bin_alloc.values()])
                new_fitting_bins = [[x.id for x in y] for k,y in new_total_bin_alloc.items() if init_len_bin_alloc <= k < self.D3]
                total_bin_alloc_list += new_fitting_bins
                column_alloc_list += [j for i in new_fitting_bins for j in i]
                # List of columns still to be allocated
                #print('New allocation', total_bin_alloc_list)
                not_allocated_valid_column_subsets = [v for k,v in new_total_bin_alloc.items() if k < init_len_bin_alloc]
                not_allocated_valid_column_subsets = [x for x in not_allocated_valid_column_subsets if x != []]
                not_allocated_superitems = [list(j.superitem_set) for i in not_allocated_valid_column_subsets for j in i]
                not_allocated_items = [list(j.item_set) for i in not_allocated_superitems for j in i]
                not_allocated_items = [j for i in not_allocated_items for j in i]

                # Not allocated items: many items identical with different tile index
                # Make tile index a list instead
                not_allocated_items_clean = []
                for nai in not_allocated_items:
                    nai_copy = copy.deepcopy(nai)
                    if any([x.layer_index == nai.layer_index for x in not_allocated_items_clean]):
                        nai_original = next((x for x in not_allocated_items_clean if x.layer_index == nai.layer_index),None)
                        nai_original.tile_index.append(nai.tile_index)
                    else:
                        nai_copy.tile_index = [nai.tile_index]
                        not_allocated_items_clean.append(nai_copy)



                #print('To be allocated', [[x.id for x in y] for y  in not_allocated_valid_column_subsets])

                #print('Allocated', [[x.layer_index_set for x in y] for y in valid_column_subsets])
                #print('To be allocated', [[x.layer_index_set for x in y] for y  in not_allocated_valid_column_subsets])
                if [] in total_bin_alloc_list:
                    print('Still to be allocated', [[x.layer_index_set for x in y] for y in new_total_bin_alloc.values()])
                    print('To be allocated', [[x.layer_index_set for x in y] for y  in not_allocated_valid_column_subsets])
                    bin_dict = {}
                    return bin_dict, "UNFEASIBLE", set(not_allocated_items)
                    breakpoint()
                bin_dict =  {}
                for i in range(self.D3):
                    column_alloc = []
                    if len(total_bin_alloc_list) > i:
                        for ii_l, l in enumerate(total_bin_alloc_list[i]):
                            lx = next((x for x in column_list if x.id == l), None)
                            column_alloc.append(lx)
                        # Sort columns by D2, the largest one at the bottom
                        column_alloc.sort(key=lambda x:x.actual_depth,reverse=True)
                        column_alloc = [x.id for x in column_alloc]
                    bin_dict[i] = column_alloc
                #print('New bin dict', bin_dict)
                return bin_dict, "UNFEASIBLE", set(not_allocated_items)

            # Remove allocated columns from the global list
            total_bin_alloc_list += new_total_bin_alloc_list
            column_alloc_list += new_column_alloc_list
            column_lite_list = [x for x in column_lite_list if x.id not in column_alloc_list]
            #pbar.update(init_range - len(column_lite_list))
            init_range = len(column_lite_list)
        #pbar.close()
        bin_dict =  {}
        for i in range(self.D3):
            column_alloc = []
            if len(total_bin_alloc_list) > i:
                for ii_l, l in enumerate(total_bin_alloc_list[i]):
                    lx = next((x for x in column_list if x.id == l), None)
                    column_alloc.append(lx)
                # Sort columns by D2, the largest one at the bottom
                column_alloc.sort(key=lambda x:x.actual_depth,reverse=True)
                column_alloc = [x.id for x in column_alloc]
            bin_dict[i] = column_alloc
        
        if self.verbose == 1:
            logger.info(f'Generated Bins #{len(total_bin_alloc_list)}')#; Layers to be assigned: {len(column_lite_list)}')
        return bin_dict, "OPTIMAL", None


    def pack_macrobin(self, column_list, fsi, zsl, ol, nki):
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        n_superitems, n_items = fsi.shape
        n_network_columns, _ = nki.shape

        # Variable definition
        tb, al, ulb = dict(), dict(), dict()
        for b in range(self.D3):
            tb[b] = model.NewBoolVar(f"t_{b}")
        for l in range(len(column_list)):
            #al[l] = model.NewBoolVar(f'a_{l}')
            for b in range(self.D3):
                ulb[l,b] = model.NewBoolVar(f'u_{l}_{b}')
        
        # Constraints
        # Ensure that every item is included in exactly one column
        for i in range(n_items):
            model.Add(
                cp_model.LinearExpr.Sum(
                    fsi[s, i] * zsl[s, l] * ulb[l, b] for s in range(n_superitems) for l in range(len(column_list)) for b in range(self.D3)
                )
                == 1
            )

        for k in range(n_network_columns):
            for b in range(self.D3):
                model.Add(
                    cp_model.LinearExpr.Sum(
                        nki[k, i] * fsi[s, i] * zsl[s, l] * ulb[l, b] for i in range(n_items) for s in range(n_superitems) for l in range(len(column_list))
                    )
                    <= 1
                )


        # Ensure that height of column combination does not exceed bin height
        for b in range(self.D3):
            model.Add(
                cp_model.LinearExpr.Sum(
                    ol[l] * ulb[l, b] for l in range(len(column_list))
                )
                <= self.height
            )
        for l in range(len(column_list)):
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
                for l in range(len(column_list)):
                    if solver.Value(ulb[l,b]) == 1:
                        bin_dict[b].append(l)
        return bin_dict, solver.StatusName(status)

        

