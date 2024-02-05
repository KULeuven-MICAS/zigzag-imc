import numpy as np
import pandas as pd
import itertools
import pickle
from item import Item


def item_pool_generator(d1,d2):

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

    items = []

    for ii_n, n in network.items():
        k_pf = [('K',x) for x in prime_factors(n['K'])]
        c_pf = [('C',x) for x in prime_factors(n['C'])]
        fx_pf = [('FX',x) for x in prime_factors(n['FX'])]
        fy_pf = [('FY',x) for x in prime_factors(n['FY'])]
        
        d1_comb = []
        d2_comb = []
        max_d1, max_d2 = 0,0

        for k in range(len(k_pf)):
            for c in itertools.combinations(k_pf,k):
                if np.prod([x[1] for x in c]) <= d1 and np.prod([x[1] for x in c]) > max_d1:
                    cx = []
                    for lpf in c:
                        if lpf[0] not in [x[0] for x in cx]:
                            cx.append(list(lpf))
                        else:
                            lpfx = next((x for x in cx if x[0] == lpf[0]),None)
                            lpfx[1] *= lpf[1]
                    cx = tuple([tuple(x) for x in cx])
                    d1_comb = cx
                    
        
        for k in range(len(c_pf) + len(fx_pf) + len(fy_pf)):
            for c in itertools.combinations(c_pf + fx_pf + fy_pf, k):
                if np.prod([x[1] for x in c]) <= d2 and np.prod([x[1] for x in c]) > max_d2:
                    cx = []
                    for lpf in c:
                        if lpf[0] not in [x[0] for x in cx]:
                            cx.append(list(lpf))
                        else:
                            lpfx = next((x for x in cx if x[0] == lpf[0]),None)
                            lpfx[1] *= lpf[1]
                    cx = tuple([tuple(x) for x in cx])
                    d2_comb = cx

        item_repetition = np.prod([x for x in n.values()]) / np.prod([x[1] for x in d1_comb + d2_comb])
        for it in range(int(item_repetition)):
            width = np.prod([x[1] for x in d1_comb])
            depth = np.prod([x[1] for x in d2_comb])

#            items.append({'width':width, 'depth':depth, 'height':1, 'volume':width*depth*1, 'network_layer':ii_n, 'weight': 1})
            items.append(Item(width=width, depth=depth, height=1, volume=width*depth*1, layer_index=ii_n, tile_index=it))

    return items

if __name__ == "__main__":
    pass
