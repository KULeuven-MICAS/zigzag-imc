from zigzag.classes.opt.CGA.item import Item, ItemPool
from zigzag.classes.opt.CGA.superitem import *
from zigzag.classes.opt.CGA.macro_bin import MacroBin
from zigzag.classes.opt.CGA.utils import plot_item_allocation
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



class CGAColumn():
    def __init__(self, width, depth):
        self.height = 0
        self.layer_index_set = set()
        self.superitem_set = set()
        self.macro_allocation = 0
        self.depth = depth
        self.width = width
        self.actual_depth = 0
        self.actual_width = 0
        self.volume = 0
        self.density = 0
        self.id = 0

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if set([x for x in self.superitem_set]) == set([x for x in other.superitem_set]):
                return True
            else:
                return False
        else:
            return False

    def __hash__(self):
        return hash((hash((hash(y) for y in x.item_set)) for x in self.superitem_set))


    def add_superitem(self, superitem):
        self.superitem_set.add(superitem)
        if superitem.height > max([x.height for x in self.superitem_set]):
            self.height = superitem.height
        self.layer_index_set.update(superitem.layer_index_set)
        if self.actual_width < superitem.x_pos + superitem.width:
            self.actual_width = superitem.x_pos + superitem.width
        if self.actual_depth < superitem.y_pos + superitem.depth:
            self.actual_depth = superitem.y_pos + superitem.depth

 
    def get_volume(self):
        volume = np.sum([x.get_volume() for x in self.superitem_set])
        return volume

    def get_total_volume(self):
        total_volume = max([x.height for x in self.superitem_set]) * self.width * self.depth
        return total_volume


    def get_area(self):
        area = np.sum([x.get_area() for x in self.superitem_set])
        return area


class ColumnPool():
    def __init__(self, D1, D2, network_layers):
        self.width = D1
        self.depth = D2
        self.superitem_index = 0
        self.item_index = 0
        self.column_index = 0
        self.n_network_layers = network_layers


    def get_volume(self, superitem_list):
        volume = np.sum([x.get_volume() for x in superitem_list])
        return volume

    def get_total_volume(self, superitem_list):
        total_volume = max([x.height for x in superitem_list]) * self.width * self.depth
        return total_volume


    @staticmethod
    def pack_2D(superitems, column):
        rectangles = []
        for ii_si, si in enumerate(superitems):
            rectangles.append((si.width, si.depth, ii_si))
        p = newPacker(bin_algo=packer.PackingBin.BFF, mode=packer.PackingMode.Offline, rotation=False)

        # Add the rectangles to packing queue
        for r in rectangles:
            p.add_rect(*r)
        # Add the bins where the rectangles will be placed
        # WIDTH = D1, DEPTH = D2
        p.add_bin(column.width, column.depth)
        # Start packing
        p.pack()
        if len(p[0]) != len(rectangles):
            return False, 0, 0
        else:
            for index, abin in enumerate(p):
                bw, bh  = abin.width, abin.height
                for ii_r in range(len(superitems)):
                    rect = next((x for x in abin if x.rid == ii_r),None)
                    x, y, w, h = rect.x, rect.y, rect.width, rect.height
                    superitems[ii_r].x_pos = x
                    superitems[ii_r].y_pos = y

            density = column.get_volume() / column.get_total_volume()
            return True, density, superitems

    def generate_cp_parameters(self):
        fsi = np.zeros((self.superitem_index, self.item_index), dtype=np.int8)
        zsl = np.zeros((self.superitem_index, self.column_index), dtype=np.int8)
        ol = np.zeros((self.column_index,), dtype=np.int64)
        nki = np.zeros((self.n_network_columns, self.item_index), dtype=np.int8)

        for ii_l, l in enumerate(self.total_column_list):
            ol[l.id] = int(l.height)
            for ii_s, s in enumerate(l.superitem_set):
                zsl[s.id, l.id] = 1
                for ii_i, i in enumerate(s.item_set):
                    fsi[s.id, i.id] = 1
                    nki[i.column_index, i.id] = 1

        return fsi, zsl, ol, nki 

        

    def generate_columns_from_comb(self, comb):
        num_columns = float('inf')
        for si in comb:
            for i in si.item_set:
                if i.tile_index < num_columns:
                    num_columns = i.tile_index
        
        comb_list = list(comb)
        column_list = []
        for n in range(1,int(num_columns)+1):
            column_new = CGAColumn(self.width, self.depth)
            for si in comb_list:
                si_new = SuperItem()
                for i in si.item_set:
                    ix = copy.deepcopy(i)
                    ix.tile_index -= n
                    ix.id = self.item_index
                    ix.x_pos = si.x_pos
                    ix.y_pos = si.y_pos
                    self.item_index += 1
                    si_new.add_item(ix)
                si_new.id = self.superitem_index
                self.superitem_index += 1
                column_new.add_superitem(si_new)
            column_new.id = self.column_index
            column_new.height = max([x.height for x in column_new.superitem_set])
            self.column_index += 1
            #print([[x for x in y.item_set] for y in column_new.superitem_set])
            column_list.append(column_new)

        return column_list

    def update_superitem_pool(self, superitem_pool, comb):
        num_columns = float('inf')
        for si in comb:
            for i in si.item_set:
                if i.tile_index < num_columns:
                    num_columns = i.tile_index

        new_si_pool = set()
        comb_items = [[x for x in y.item_set] for y in comb]
        comb_items = [j for i in comb_items for j in i]
        for si in superitem_pool:
            si_new = SuperItem()
            tiles_to_be_allocated_still = True
            for i in si.item_set:
                if i in comb_items:
                    ix = copy.deepcopy(i)
                    ix.tile_index -= num_columns
                    if ix.tile_index <= 0:
                        tiles_to_be_allocated_still = False
                        break
                    si_new.add_item(ix)
            if tiles_to_be_allocated_still and si_new.item_set != set():
                new_si_pool.add(si_new)

        for si in superitem_pool:
            if all([x not in comb_items for x in si.item_set]):
                new_si_pool.add(si)

        return new_si_pool


    @staticmethod
    def column_generate_recursive(column, superitem_pool, column_list):
        if superitem_pool == set():
            column.density = column.get_volume() / column.get_total_volume()
            column_list.append(column)
        else:
            for superitem in superitem_pool:
                if superitem.layer_index_set.intersection(column.layer_index_set) != set():
                    ColumnPool.column_generate_recursive(column, superitem_pool=set(), column_list=column_list)
                else:
                    superitem_set = copy.deepcopy(column.superitem_set)
                    superitem_set.add(copy.deepcopy(superitem))
                    fitting, density, superitems_comb = ColumnPool.pack_2D(list(superitem_set), column)
                    if not fitting:
                        ColumnPool.column_generate_recursive(column, superitem_pool=set(), column_list=column_list)
                    else:    
                        column_copy = CGAColumn(column.width, column.depth)
                        for sic in superitems_comb:
                            column_copy.add_superitem(sic)
                        column_copy.density = density
                        superitem_pool_copy = copy.deepcopy(superitem_pool)
                        superitem_pool_copy = set([x for x in superitem_pool_copy if x != superitem])
                        ColumnPool.column_generate_recursive(column_copy, superitem_pool_copy, column_list)


    def generate(self, superitem_pool):
        total_column_list = []
        while superitem_pool != set():
            max_density = 0
            for ii_si, si in enumerate(superitem_pool):
                column_list_si = []
                column = CGAColumn(self.width, self.depth)
                column.add_superitem(si)
                si_pool_copy = copy.deepcopy(superitem_pool)
                si_pool_copy = set([x for x in si_pool_copy if x != si])
                ColumnPool.column_generate_recursive(column, si_pool_copy, column_list_si)
                for column in column_list_si:
                    if column.density > max_density:
                        max_density = column.density
                        best_comb = copy.deepcopy(column.superitem_set)

            column_list = self.generate_columns_from_comb(best_comb)
            total_column_list += column_list
            superitem_pool = self.update_superitem_pool(superitem_pool, best_comb)
#            logger.info(f'Generated Layer #{len(total_column_list)}; SuperItems to be assigned: {len(superitem_pool)}')

        self.total_column_list = total_column_list
#        logger.info(f'Generated Layers #{len(total_column_list)}')
        return total_column_list



if __name__ == "__main__":

    # LAYERS MUST START FROM INDEX ZERO!
    network = {0:{'K':16,  'C':3,   'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               1:{'K':16,  'C':16,  'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               2:{'K':16,  'C':16,  'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               3:{'K':32,  'C':16,  'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               4:{'K':32,  'C':32,  'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               5:{'K':32,  'C':16,  'FX':1,'FY':1, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               6:{'K':64,  'C':32,  'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               7:{'K':64,  'C':64,  'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               8:{'K':64,  'C':32,  'FX':1,'FY':1, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               9:{'K':10,  'C':64,  'FX':1,'FY':1, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1}}

    solver_status = ""
    M = 128
    D1 = 32
    D2 = 32 
    D3 = 4

    itempool = ItemPool(D1=D1,D2=D2,D3=D3,M=M,network=network)
    itempool.set_init_M()
    while solver_status not in ['OPTIMAL','FEASIBLE']:
        logger.info("===== ItemPool Generation =====")
        item_pool, feasible_tile_configuration, target_column_index = itempool.generate()
        while not feasible_tile_configuration:
            itempool.update_network(target_column_index)
            item_pool, feasible_tile_configuration, target_column_index = itempool.generate()
        si = SuperItemPool(item_pool)
        logger.success("===== ItemPool Generation Done =====")
        logger.info("===== SuperItemPool Generation =====")
        superitem_pool = si.generate()
        logger.info("===== ColumnPool Generation =====")
        column_pool = ColumnPool(D1=D1,D2=D2, network_columns=len(network.keys()))
        column_list = column_pool.generate(superitem_pool)
        fsi, zsl, ol, nki = column_pool.generate_cp_parameters()
        macro_bin = MacroBin(height=M, number_of_macros=D3)
        bin_dict, solver_status = macro_bin.pack_macrobin(column_list, fsi, zsl, ol, nki)
        if solver_status in ['OPTIMAL','FEASIBLE']:
            plot_item_allocation(column_list, bin_dict, D3=D3, height=M,D1=D1,D2=D2)
            logger.success('>>>> Completed allocation <<<<')
            break
        else:
            feasible_configuration = itempool.update_network()

        if not feasible_configuration:
            logger.error('>>>> Unfeasible settings for D1,D2,D3,M <<<<')
            break
