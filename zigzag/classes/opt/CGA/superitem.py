import numpy as np
import itertools
from item import Item
import copy


class SuperItem():
    def __init__(self, height=0, width=0, depth=0):
        self.height = height
        self.depth = depth
        self.width = width
        self.layer_index_set = set()
        self.item_set = set()
        self.base_item = None

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


    def add_item(self, item):
        if self.item_set == set():
            self.base_item = item
        self.item_set.add(item)
        self.height += item.height
        if item.depth > max([x.depth for x in self.item_set]):
            self.depth = item.depth
        if item.width > max([x.width for x in self.item_set]):
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
                elif item.width > superitem.width or item.length > superitem.length:
                    SuperItemPool.superitem_generate_recursive(superitem, item_pool=set(), max_height=max_height, superitem_list=superitem_list)
                else:
                    superitem_copy = copy.deepcopy(superitem)
                    superitem_copy.add_item(item)
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
        return set(superitem_list)


class SuperItemSetPool():
    def __init__(self, superitem_pool):
        self.superitem_pool = copy.deepcopy(superitem_pool)
        self.superitem_set_pool = []
        self.height_list = np.unique([x.height for x in superitem_pool])
        self.superitem_pool_unique_items = []
        for si in self.superitem_pool:
            if all([x.tile_index == 0 for x in si.item_set]):
                self.superitem_pool_unique_items.append(si)
        self.superitem_pool_unique_items = set(self.superitem_pool_unique_items)

    @staticmethod
    def superitem_set_generate_recursive(superitem_set, superitem_pool, max_area, superitem_set_list):
        if superitem_pool == set():
            if superitem_set not in superitem_set_list:
                superitem_set_list.append(superitem_set)
                print([[x for x in y.item_set] for y in superitem_set.superitem_set])
                
        else:
            for superitem in superitem_pool:
                if superitem_set.get_area() + superitem.get_area() > max_area:
                    SuperItemSetPool.superitem_set_generate_recursive(superitem_set, superitem_pool=set(), max_area=max_area, superitem_set_list = superitem_set_list)

                elif superitem.layer_index_set.intersection(superitem_set.layer_index_set) != set():
                    SuperItemSetPool.superitem_set_generate_recursive(superitem_set, superitem_pool=set(), max_area=max_area, superitem_set_list = superitem_set_list)

                elif superitem.height > superitem_set.height:
                    SuperItemSetPool.superitem_set_generate_recursive(superitem_set, superitem_pool=set(), max_area=max_area, superitem_set_list = superitem_set_list)

                else:
                    superitem_set_copy = copy.deepcopy(superitem_set)
                    superitem_set_copy.add_superitem(superitem)
                    superitem_pool_copy = set([x for x in superitem_pool if x != superitem])
                    SuperItemSetPool.superitem_set_generate_recursive(superitem_set_copy, superitem_pool_copy, max_area, superitem_set_list)


    def generate(self, max_area=0):
        superitem_set_list = []
        for superitem in self.superitem_pool_unique_items:
            breakpoint()
            sis = SuperItemSet()
            sis.add_superitem(superitem)
            print(sis)
            superitem_pool_copy = copy.deepcopy(self.superitem_pool_unique_items)
            superitem_pool_copy = set([x for x in superitem_pool_copy if x != superitem])
            SuperItemSetPool.superitem_set_generate_recursive(sis, superitem_pool_copy, max_area, superitem_set_list)
        self.superitem_set_list = set(superitem_set_list)
        return set(superitem_set_list)













