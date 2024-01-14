import itertools
import numpy as np
import random
import math
import pickle

with open('cacti_table.pkl','rb') as infile:
    mem_data = pickle.load(infile)

breakpoint()


aa = 46733904322560000
hh = random.sample(range(0,aa),10000)
breakpoint()
def get_adder_depth_structure(adder_depth):
    adder_depth_structure = []
    while adder_depth > 0:
        print(adder_depth, math.floor(math.log2(adder_depth)))
        adder_depth_structure.append(2**math.floor(math.log2(adder_depth)))
        adder_depth -= 2**(math.floor(math.log2(adder_depth)))

    return adder_depth_structure

aa = get_adder_depth_structure(24)
print(aa)
breakpoint()
