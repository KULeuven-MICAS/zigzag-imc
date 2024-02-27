from zigzag.classes.opt.CGA.item import *
from zigzag.classes.opt.CGA.superitem import *
from zigzag.classes.opt.CGA.column import *
from zigzag.classes.opt.CGA.macro_bin import *
import itertools
from shapely.geometry import Polygon
import shapely


class RewriteAllocation():
    def __init__(self, bin_dict_allocated, column_list, not_allocated_item_pool, network, D1, D2, D3, M):
        self.bin_dict_allocated = bin_dict_allocated
        self.column_list = column_list
        self.not_allocated_item_pool = not_allocated_item_pool
        self.network = network
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.M = M


    def check_overlap_of_cuboids(self, item_al, item_nal):
        """
                 [8]------[7]    
                 /|       /|
                / |      / |
               /  |     /  |
            [4]--[5]-[3]--[6]
             |   /    |   /
             |  /     |  /
             | /      | /
            [1]------[2]

        """

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
        overlap_rows_cuboid = [ max(item_al.y_pos, item_nal.y_pos), min(item_al.y_pos + item_al.depth, item_nal.y_pos + item_nal.depth),\
            max(item_al.z_pos, item_nal.z_pos), min(item_al.z_pos + item_al.height, item_nal.z_pos + item_nal.height)]
         
        if overlap_volume > 0:
            overlap_cuboid = [min(item_al.x_pos + item_al.width, item_nal.x_pos + item_nal.width),max(item_al.x_pos, item_nal.x_pos),
                min(item_al.y_pos + item_al.depth, item_nal.y_pos + item_nal.depth), max(item_al.y_pos, item_nal.y_pos),
                min(item_al.z_pos + item_al.height, item_nal.z_pos + item_nal.height), max(item_al.z_pos, item_nal.z_pos)]
            overlap_rect = [min(item_al.y_pos + item_al.depth, item_nal.y_pos + item_nal.depth), max(item_al.y_pos, item_nal.y_pos),
                min(item_al.z_pos + item_al.height, item_nal.z_pos + item_nal.height), max(item_al.z_pos, item_nal.z_pos)]

            return overlap_cuboid, overlap_volume, overlap_rect, overlap_area
        else:
            return None, None, None, None


    def compute_overlap_volume(self, comb):
        item1 = comb[0]
        item2 = comb[1]
        volume = max(min(item1[0], item2[0]) - max(item1[1], item2[1]),0)
        volume*= max(min(item1[2], item2[2]) - max(item1[3], item2[3]),0)
        volume*= max(min(item1[4], item2[4]) - max(item1[5], item2[5]),0)
        return volume


    def compute_overlap_area(self, comb):
        item1 = comb[0]
        item2 = comb[1]
        area = max(min(item1[0], item2[0]) - max(item1[1], item2[1]),0)
        area*= max(min(item1[2], item2[2]) - max(item1[3], item2[3]),0)
        return area


    def generate_item_dict_with_positions(self, bin_dict, bin_dict_new_alloc, column_list, D2, M):
        # TODO ALGORITHM TO FIND OPTIMAL ALLOCATION OF ITEMS SO AS TO MINIMIZE OVERLAP ACROSS REWRITES
        # Computes the number of cells, rows required to be rewritten in case not all items are allocated
        bin_dict_items = {}
        for k,v in bin_dict.items():
            column_bins = [next((x for x in column_list if x.id == l),None) for l in v]
            bin_dict_new_alloc[k].insert(0,column_bins)

        for k,v in bin_dict_new_alloc.items():
            # v is a list of columns; each v is a weight writing cycle
            bin_dict_items[k] = [[] for x in v]
            for ii_c, column_list_in_bin in enumerate(v):
                if ii_c == 0:
                    column_list_in_bin.sort(key=lambda x:x.depth, reverse=True)
                    base_z = 0
                else:
                    column_list_in_bin.sort(key=lambda x:x.depth)
                    base_z = M - sum([x.height for x in column_list_in_bin])
                for column in column_list_in_bin:
                    for superitem in column.superitem_set:
                        base_z_si = base_z
                        for item in superitem.item_set:
                            # Check if item overlaps with allocated items
                            item.x_pos = superitem.x_pos
                            if ii_c == 0:
                                item.y_pos = superitem.y_pos
                            else:
                                item.y_pos = D2 - superitem.depth
                            item.z_pos = base_z_si
                            bin_dict_items[k][ii_c].append(item)
                            base_z_si += item.height
                    base_z += column.height
        return bin_dict_items


    def weight_reloading_cells(self, column_list, not_allocated_columns, bin_dict, D2, M, network, bin_dict_new_alloc):
        # Create list with all items PER WEIGHT WRITING CYCLE, PER BIN and their positions
        bin_dict_items = self.generate_item_dict_with_positions(bin_dict, bin_dict_new_alloc, column_list, D2, M)
        overlapping_areas = {}
        total_overlap_volume_layer = {k:0 for k in network.keys()}
        total_overlap_area_layer = {k:0 for k in network.keys()}

        for k,v in bin_dict_items.items():
            overlapping_areas[k] = {} 
            for ii_wr, column_list_bin in enumerate(bin_dict_items[k]):
#                overlapping_volumes[k].append({})
#                overlapping_areas[k].append({})
                other_items = bin_dict_items[k][:ii_wr] + bin_dict_items[k][ii_wr:]
                other_items = [j for i in other_items for j in i]
                for item in bin_dict_items[k][ii_wr]:
                    overlapping_areas[k][item] = []
                    for oitem in other_items:
                        # check if item and oitem overlap
                        overlap_cuboid, overlap_volume, overlap_rect, overlap_area = self.check_overlap_of_cuboids(item, oitem)                
                        # if they overlap, add overlap coordinates to list
                        if overlap_cuboid != None:
                            oc = overlap_rect
                            overlapping_areas[k][item].append(Polygon([(oc[1],oc[3]),(oc[0],oc[3]),(oc[0],oc[2]),(oc[1],oc[2])]))
                    # remove "overlapping overlaps" from total_volume, total_area
                for item in bin_dict_items[k][ii_wr]:
                    item_rect = Polygon([(item.y_pos, item.z_pos),(item.y_pos + item.depth, item.z_pos),(item.y_pos + item.depth, item.z_pos + item.height),(item.y_pos, item.z_pos+item.height)])
                    for oitem in bin_dict_items[k][ii_wr]:
                        if oitem == item:
                            continue
                        for ov_polygon in overlapping_areas[k][oitem]:
                            if item_rect.intersects(ov_polygon):
                                overlapping_areas[k][item].append(item_rect.intersection(ov_polygon))
                for item in bin_dict_items[k][ii_wr]:
                    p = Polygon([])
                    for ov_item in overlapping_areas[k][item]:
                        p = shapely.union(p, ov_item)

                    # p.area is the total number of rows to be rewritten
                    # item.width is the number of cells per row to be rewritten
                    total_overlap_volume_layer[item.layer_index] += p.area * item.width
                    total_overlap_area_layer[item.layer_index] += p.area



        return total_overlap_volume_layer, total_overlap_area_layer
            
        

    def run(self):
        not_allocated_si = SuperItemPool(self.not_allocated_item_pool)
        not_allocated_superitem_pool = not_allocated_si.generate()
        not_allocated_column_pool = ColumnPool(D1=self.D1,D2=self.D2,network_layers=len(self.network.keys()))

        # Create a list of lists for each bin: each sublist corresponds to a column set allocation that overlaps with previous ones --> weight rewriting is required
        # Eg {0 : [[Col1], [Col2]], 1: [[Col3, Col4]]} --> for bin 0, Col1 and Col2 have to be written in separate occasions, while Col3 and Col4 can be allocated in the same bin 1
        not_allocated_column_list, bin_dict_weight_rewriting = not_allocated_column_pool.generate_not_allocated(not_allocated_superitem_pool, self.bin_dict_allocated, self.column_list, self.M)
        extra_cells, extra_rows = self.weight_reloading_cells(self.column_list, not_allocated_column_list, self.bin_dict_allocated, self.D2, self.M, self.network, bin_dict_weight_rewriting)

        extra_cells, extra_rows = self.convert_layer_index_to_workload(extra_cells, extra_rows, self.network)
        return extra_cells, extra_rows

    def convert_layer_index_to_workload(self, extra_cells, extra_rows, network):
        extra_cells_new = {}
        extra_rows_new = {}

        for k,v in extra_cells.items():
            extra_cells_new[network[k]['layer_id']] = v
        for k,v in extra_rows.items():
            extra_rows_new[network[k]['layer_id']] = v

        return extra_cells_new, extra_rows_new


