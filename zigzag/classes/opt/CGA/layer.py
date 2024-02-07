from item import Item, ItemPool
from superitem import *
from macro_bin import MacroBin
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
from utils import plot_item_allocation



class CGALayer():
    def __init__(self):
        self.height = 0
        self.layer_index_set = set()
        self.superitem_set = set()
        self.macro_allocation = 0
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
 
    def get_volume(self):
        volume = np.sum([x.get_volume() for x in self.superitem_set])
        return volume

    def get_area(self):
        area = np.sum([x.get_area() for x in self.superitem_set])
        return area


class LayerPool():
    def __init__(self, D1, D2, network_layers):
        self.width = D1
        self.depth = D2
        self.superitem_index = 0
        self.item_index = 0
        self.layer_index = 0
        self.n_network_layers = network_layers


    def get_volume(self, superitem_list):
        volume = np.sum([x.get_volume() for x in superitem_list])
        return volume

    def get_total_volume(self, superitem_list):
        total_volume = max([x.height for x in superitem_list]) * self.width * self.depth
        return total_volume


    def pack_2D(self, superitems):
        rectangles = []
        for ii_si, si in enumerate(superitems):
            rectangles.append((si.width, si.depth, ii_si))
        p = newPacker(bin_algo=packer.PackingBin.BFF, mode=packer.PackingMode.Offline, rotation=False)

        # Add the rectangles to packing queue
        for r in rectangles:
            p.add_rect(*r)
        # Add the bins where the rectangles will be placed
        # WIDTH = D1, DEPTH = D2
        p.add_bin(self.width, self.depth)
        # Start packing
        p.pack()
        if len(p[0]) != len(rectangles):
            return False, 0, 0
        else:
            for index, abin in enumerate(p):
                bw, bh  = abin.width, abin.height
            #    print('bin', bw, bh, "nr of rectangles in bin", len(abin))
            #    fig = plt.figure()
            #    ax = fig.add_subplot(111, aspect='equal')
                for ii_r in range(len(superitems)):
                    rect = next((x for x in abin if x.rid == ii_r),None)
                    x, y, w, h = rect.x, rect.y, rect.width, rect.height
                    superitems[ii_r].x_pos = x
                    superitems[ii_r].y_pos = y
            #        plt.axis([0,bw,0,bh])
            #        print('rectangle', w,h)
            #        ax.add_patch(
            #            patches.Rectangle(
            #                (x, y),  # (x,y)
            #                w,          # width
            #                h,          # height
            #                facecolor="#00ffff",
            #                edgecolor="black",
            #                linewidth=3
            #            )
            #        )
            #    fig.savefig("rect_%(index)s.png" % locals(), dpi=144, bbox_inches='tight')
            density = self.get_volume(superitems) / self.get_total_volume(superitems)
            return True, density, superitems

    def generate_cp_parameters(self):
        fsi = np.zeros((self.superitem_index, self.item_index), dtype=np.int8)
        zsl = np.zeros((self.superitem_index, self.layer_index), dtype=np.int8)
        ol = np.zeros((self.layer_index,), dtype=np.int64)
        nki = np.zeros((self.n_network_layers, self.item_index), dtype=np.int8)

        for ii_l, l in enumerate(self.total_layer_list):
            ol[l.id] = int(l.height)
            for ii_s, s in enumerate(l.superitem_set):
                zsl[s.id, l.id] = 1
                for ii_i, i in enumerate(s.item_set):
                    fsi[s.id, i.id] = 1
                    nki[i.layer_index, i.id] = 1

        return fsi, zsl, ol, nki 

        

    def generate_layers_from_comb(self, comb):
        num_layers = float('inf')
        for si in comb:
            for i in si.item_set:
                if i.tile_index < num_layers:
                    num_layers = i.tile_index
        
        comb_list = list(comb)
        layer_list = []
        for n in range(1,int(num_layers)+1):
            layer_new = CGALayer()
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
                layer_new.add_superitem(si_new)
            layer_new.id = self.layer_index
            layer_new.height = max([x.height for x in layer_new.superitem_set])
            self.layer_index += 1
            #print([[x for x in y.item_set] for y in layer_new.superitem_set])
            layer_list.append(layer_new)

        return layer_list

    def update_superitem_pool(self, superitem_pool, comb):
        num_layers = float('inf')
        for si in comb:
            for i in si.item_set:
                if i.tile_index < num_layers:
                    num_layers = i.tile_index

        new_si_pool = set()
        comb_items = [[x for x in y.item_set] for y in comb]
        comb_items = [j for i in comb_items for j in i]
        for si in superitem_pool:
            si_new = SuperItem()
            tiles_to_be_allocated_still = True
            for i in si.item_set:
                if i in comb_items:
                    ix = copy.deepcopy(i)
                    ix.tile_index -= num_layers
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



    def generate(self, superitem_pool):
        total_layer_list = []
        while superitem_pool != set():
            max_density = 0
            best_comb = None
            for k in range(len(superitem_pool),0,-1):
                k_fitted = False
                for comb in itertools.combinations(superitem_pool, k):
                    if comb == None:
                        break 
                    # check that the combination contains at most
                    # 1 item per layer
                    layer_set_count = []
                    for si in comb:
                        layer_set_count += list(si.layer_index_set)
                    cnt = Counter(layer_set_count)
                    if any([x > 1 for k,x in cnt.items()]):
                        continue
                    fitting, density, superitems_comb = self.pack_2D(comb)
                    if density > max_density:
                        max_density = density
                        best_comb = copy.deepcopy(superitems_comb)
                        k_fitted = True
                if k_fitted == False and best_comb is not None:
                    break

            layer_list = self.generate_layers_from_comb(best_comb)
            total_layer_list += layer_list
            superitem_pool = self.update_superitem_pool(superitem_pool, best_comb)
            logger.info(f'Generated Layer #{len(total_layer_list)}; SuperItems to be assigned: {len(superitem_pool)}')

        self.total_layer_list = total_layer_list
        return total_layer_list



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
        item_pool, feasible_tile_configuration, target_layer_index = itempool.generate()
        while not feasible_tile_configuration:
            itempool.update_network(target_layer_index)
            item_pool, feasible_tile_configuration, target_layer_index = itempool.generate()
        si = SuperItemPool(item_pool)
        logger.success("===== ItemPool Generation Done =====")
        logger.info("===== SuperItemPool Generation =====")
        superitem_pool = si.generate()
        logger.info("===== LayerPool Generation =====")
        layer_pool = LayerPool(D1=D1,D2=D2, network_layers=len(network.keys()))
        layer_list = layer_pool.generate(superitem_pool)
        fsi, zsl, ol, nki = layer_pool.generate_cp_parameters()
        macro_bin = MacroBin(height=M, number_of_macros=D3)
        bin_dict, solver_status = macro_bin.pack_macrobin(layer_list, fsi, zsl, ol, nki)
        if solver_status in ['OPTIMAL','FEASIBLE']:
            plot_item_allocation(layer_list, bin_dict, D3=D3, height=M,D1=D1,D2=D2)
            logger.success('>>>> Completed allocation <<<<')
            break
        else:
            feasible_configuration = itempool.update_network()

        if not feasible_configuration:
            logger.error('>>>> Unfeasible settings for D1,D2,D3,M <<<<')
            break
