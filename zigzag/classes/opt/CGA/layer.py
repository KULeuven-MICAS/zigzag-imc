from item import *
from superitem import *
import numpy as np
import itertools
import copy


class CGALayer():
    def __init__(self):
        pass

if __name__ == "__main__":

    network = {1:{'K':16,  'C':3,   'FX':3,'FY':3},
               2:{'K':16,  'C':16,  'FX':3,'FY':3},
               3:{'K':16,  'C':16,  'FX':3,'FY':3},
               4:{'K':32,  'C':16,  'FX':3,'FY':3},
               5:{'K':32,  'C':32,  'FX':3,'FY':3},
               6:{'K':32,  'C':16,  'FX':1,'FY':1},
               7:{'K':64,  'C':32,  'FX':3,'FY':3},
               8:{'K':64,  'C':64,  'FX':3,'FY':3},
               9:{'K':64,  'C':32,  'FX':1,'FY':1},
              10:{'K':10,  'C':64,  'FX':1,'FY':1}}


    item_pool = Item.item_pool_generator(100,100,network)
    si = SuperItemPool(item_pool)
    superitem_pool = si.generate()
    sis = SuperItemSetPool(superitem_pool)
    superitemset_pool = sis.generate(100 * 100)

    breakpoint()

