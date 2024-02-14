import itertools
import numpy as np
import random
import math
import pickle
from rtree import index

idx = index.Index()
idx.insert(0,(0.,0.,1.,1.))
gg = list(idx.intersection((0.5,0.5,2.,2.)))
breakpoint()
#with open('cacti_table.pkl','rb') as infile:
#    mem_data = pickle.load(infile)
#
#breakpoint()
#
#
#aa = 46733904322560000
#hh = random.sample(range(0,aa),10000)
#breakpoint()
#def get_adder_depth_structure(adder_depth):
#    adder_depth_structure = []
#    while adder_depth > 0:
#        print(adder_depth, math.floor(math.log2(adder_depth)))
#        adder_depth_structure.append(2**math.floor(math.log2(adder_depth)))
#        adder_depth -= 2**(math.floor(math.log2(adder_depth)))
#
#    return adder_depth_structure
#
#aa = get_adder_depth_structure(24)
#print(aa)
#breakpoint()
