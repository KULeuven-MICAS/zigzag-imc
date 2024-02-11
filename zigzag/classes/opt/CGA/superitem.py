import numpy as np
import itertools
from zigzag.classes.opt.CGA.item import Item
import copy
from loguru import logger

class SuperItem():
    def __init__(self, height=0, width=0, depth=0):
        self.height = height
        self.depth = depth
        self.width = width
        self.layer_index_set = set()
        self.item_set = set()
        self.base_item = None
        self.x_pos = 0
        self.y_pos = 0
        self.id = 0

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.item_set == other.item_set:
                return True
            else:
                return False
        else:
            return False

    def __hash__(self):
        return hash((hash(x) for x in self.item_set))
    
    def __repr__(self):
        return f'SuperItem: Base: {self.base_item}, Set:{[x for x in self.item_set]}'


    def add_item(self, item):
        if self.item_set == set():
            self.base_item = item
        self.item_set.add(item)
        self.height += item.height
        if item.depth >= max([x.depth for x in self.item_set]):
            self.depth = item.depth
        if item.width >= max([x.width for x in self.item_set]):
            self.width = item.width
        self.layer_index_set.add(item.layer_index)
        
    def get_volume(self):
        volume = 0
        for item in self.item_set:
            volume += item.volume
        return volume

    def get_area(self):
        area = max([x.area for x in self.item_set])
        return area


class SuperItemSet():
    # set of superitems that do not contain items from more than one layer
    # and whose size does not exceed D1 x D2
    # The set is used for the 2D bin packing process
    def __init__(self):
        self.height = 0
        self.width = 0
        self.depth = 0
        self.layer_index_set = set()
        self.superitem_set = set()
        self.area = 0
        self.base_item = None
    
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
        self.area += superitem.get_area()
        if superitem.height > max([x.depth for x in self.superitem_set]):
            self.height = superitem.height
        self.layer_index_set.update(superitem.layer_index_set)
 
    def get_volume(self):
        volume = np.sum([x.get_volume() for x in self.superitem_set])
        return volume

    def get_area(self):
        area = np.sum([x.get_area() for x in self.superitem_set])
        return area

class SuperItemPool():
    def __init__(self, item_pool):
        self.item_pool = copy.deepcopy(item_pool)
        self.superitem_pool = []
        self.height_list = np.unique([x.height for x in item_pool])


    @staticmethod
    def superitem_generate_recursive(superitem, item_pool, max_height, superitem_list):
        if item_pool == set():
            if superitem not in superitem_list:
                superitem_list.append(superitem)
        else:
            for item in item_pool:
                if superitem.height + item.height > max_height:
                    SuperItemPool.superitem_generate_recursive(superitem, item_pool=set(), max_height=max_height, superitem_list=superitem_list)
                elif item.layer_index in superitem.layer_index_set:
                    SuperItemPool.superitem_generate_recursive(superitem, item_pool=set(), max_height=max_height, superitem_list=superitem_list)
                elif item.width > superitem.width or item.depth > superitem.depth:
                    SuperItemPool.superitem_generate_recursive(superitem, item_pool=set(), max_height=max_height, superitem_list=superitem_list)
                else:
                    superitem_copy = copy.deepcopy(superitem)
                    superitem_copy.add_item(item)
                    for si in superitem_list:
                        if all([x in si.layer_index_set for x in superitem_copy.layer_index_set]):
                            return
                    item_pool_copy = copy.deepcopy(item_pool)
                    item_pool_copy = set([x for x in item_pool_copy if x != item])
                    SuperItemPool.superitem_generate_recursive(superitem_copy, item_pool_copy, max_height, superitem_list)


    def generate(self):
        superitem_list = []
        for height in self.height_list:
            for item in self.item_pool:
                if item.height > height:
                    continue
                si = SuperItem()
                si.add_item(item)
                item_pool_copy = copy.deepcopy(self.item_pool)
                item_pool_copy = set([x for x in item_pool_copy if x != item])
                SuperItemPool.superitem_generate_recursive(si, item_pool_copy, height, superitem_list)
        # keep superitems with same base item and with same height that have largest volume
        optimal_superitem_list = []
        for height in self.height_list:
#            logger.info(f'Generating SuperItems for height {height}')
            si_list = [x for x in superitem_list if x.height == height]
            unique_base_items = set([x.base_item for x in si_list])
            for bi in unique_base_items:
                max_volume = 0
                optimal_ubsi_list = []
                ubsi_list = [x for x in si_list if x.base_item == bi]
                for ubsi in ubsi_list:
                    if ubsi.get_volume() > max_volume:
                        optimal_ubsi_list = [ubsi]
                        max_volume = ubsi.get_volume()
                    elif ubsi.get_volume() == max_volume:
                        optimal_ubsi_list.append(ubsi)
                optimal_superitem_list += optimal_ubsi_list
                

        self.superitem_list = set(optimal_superitem_list)
        #logger.info(f'Generated SuperItems #{len(optimal_superitem_list)}')
        return set(optimal_superitem_list)









