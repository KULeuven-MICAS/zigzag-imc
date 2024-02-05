import numpy as np
import itertools
from utils import prime_factors


class Item():
    # D1 = depth
    # D2 = width
    def __init__(self, *, height, width, depth, layer_index, tile_index):
        self.height = height
        self.depth = depth
        self.width = width
        self.layer_index = layer_index
        self.tile_index = tile_index
        self.volume = height * width * height
        self.area = width * depth

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.height == other.height and self.depth == other.depth and self.width == other.width and \
                       self.layer_index == other.layer_index and self.tile_index == other.tile_index:
                return True
            else:
                return False
        else:
            return False

    def __hash__(self):
        return hash((self.height, self.depth, self.width ,self.layer_index, self.tile_index))

    def __repr__(self):
        return f'Item L {self.layer_index} T {self.tile_index} {self.depth}x{self.width}x{self.height}'

    @staticmethod
    def item_pool_generator(d1,d2, network, m_factors=None):
        items = []

        for ii_n, n in network.items():
            k_pf = [('K',x) for x in prime_factors(n['K'])]
            c_pf = [('C',x) for x in prime_factors(n['C'])]
            fx_pf = [('FX',x) for x in prime_factors(n['FX'])]
            fy_pf = [('FY',x) for x in prime_factors(n['FY'])]
            
            d1_comb = []
            d2_comb = []
            max_d1, max_d2 = 0,0

            for k in range(len(k_pf)+1):
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
                        
            
            for k in range(len(c_pf) + len(fx_pf) + len(fy_pf) + 1):
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
                items.append(Item(width=width, depth=depth, height=1, layer_index=ii_n, tile_index=it))

        return set(items)


