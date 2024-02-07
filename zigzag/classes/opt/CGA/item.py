import numpy as np
import itertools
from utils import prime_factors
from loguru import logger

class Item():
    # D1 = width
    # D2 = depth
    def __init__(self, *, height, width, depth, layer_index, tile_index):
        self.height = height
        self.depth = depth
        self.width = width
        self.layer_index = layer_index
        self.tile_index = tile_index
        self.volume = height * width * depth
        self.area = width * depth
        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0
        self.id = 0

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
        return f'Item L{self.layer_index}T{self.tile_index} {self.depth}x{self.width}x{self.height}'


class ItemPool():
    def __init__(self, D1, D2, network, D3, M):
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.M = M
        self.network = network
        self.item_pool = None

    def generate(self):
        items = []
        feasible_tile_configuration = True
        for ii_n, n in self.network.items():
            k_pf = [('K',x) for x in prime_factors(n['K'])]
            c_pf = [('C',x) for x in prime_factors(n['C'])]
            fx_pf = [('FX',x) for x in prime_factors(n['FX'])]
            fy_pf = [('FY',x) for x in prime_factors(n['FY'])]
            
            d1_comb = []
            d2_comb = []
            max_d1, max_d2 = 0,0

            for k in range(len(k_pf)+1):
                for c in itertools.combinations(k_pf,k):
                    if np.prod([x[1] for x in c]) <= self.D1 and np.prod([x[1] for x in c]) > max_d1:
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
                    if np.prod([x[1] for x in c]) <= self.D2 and np.prod([x[1] for x in c]) > max_d2:
                        cx = []
                        for lpf in c:
                            if lpf[0] not in [x[0] for x in cx]:
                                cx.append(list(lpf))
                            else:
                                lpfx = next((x for x in cx if x[0] == lpf[0]),None)
                                lpfx[1] *= lpf[1]
                        cx = tuple([tuple(x) for x in cx])
                        d2_comb = cx
            item_repetition = np.prod([x for k,x in n.items() if k in ['K','FX','FY','C']]) / np.prod([x[1] for x in d1_comb + d2_comb])
            if item_repetition > self.D3:
                feasible_tile_configuration = False
                return None, feasible_tile_configuration, ii_n
        #    for it in range(int(item_repetition)):
            width = np.prod([x[1] for x in d1_comb])
            depth = np.prod([x[1] for x in d2_comb])
#            items.append({'width':width, 'depth':depth, 'height':1, 'volume':width*depth*1, 'network_layer':ii_n, 'weight': 1})
            if int(item_repetition) == 0:
                breakpoint()
            items.append(Item(width=int(width), depth=int(depth), height=int(n['M']), layer_index=ii_n, tile_index=int(item_repetition)))
            logger.info(f"Generated #{len(items):4} {items[-1]}")

        self.item_pool = set(items)
        return set(items), feasible_tile_configuration, None


    def set_init_M(self):
        for layer_index, layer in self.network.items():
            fitting = False
            while not fitting:
                if np.prod([layer['FX'] * layer['FY'] * layer['C']]) <= self.D2 * self.D3:
                    fitting = True
                    continue
                c_pf = [('C',x) for x in prime_factors(layer['C'])]
                fx_pf = [('FX',x) for x in prime_factors(layer['FX'])]
                fy_pf = [('FY',x) for x in prime_factors(layer['FY'])]
                
                pf = c_pf + fx_pf + fy_pf  
                pf.sort(key = lambda x: x[1])
                pf_cut = pf[0]
                self.network[layer_index][pf_cut[0]] /= pf_cut[1]
                self.network[layer_index][f'{pf_cut[0]}t'] *= pf_cut[1]
                self.network[layer_index][f'M'] *= pf_cut[1]
                logger.info(f'Network update: Layer {layer_index} cut {pf_cut[0]} by {pf_cut[1]} --> {pf_cut[0]}: {self.network[layer_index][pf_cut[0]]} {pf_cut[0]}t: {self.network[layer_index][pf_cut[0]+"t"]}, M:{self.network[layer_index]["M"]}')

        for layer_index, layer in self.network.items():
            fitting = False
            while not fitting:
                if np.prod([layer['K']]) / self.D1 <= self.D3:
                    fitting = True
                    continue
                k_pf = [('K',x) for x in prime_factors(layer['K'])]
                pf = k_pf
                pf.sort(key = lambda x: x[1])
                pf_cut = pf[0]
                self.network[layer_index][pf_cut[0]] /= pf_cut[1]
                self.network[layer_index][f'{pf_cut[0]}t'] *= pf_cut[1]
                self.network[layer_index][f'M'] *= pf_cut[1]
                logger.info(f'Network init: Layer {layer_index} cut {pf_cut[0]} by {pf_cut[1]} --> {pf_cut[0]}: {self.network[layer_index][pf_cut[0]]} {pf_cut[0]}t: {self.network[layer_index][pf_cut[0]+"t"]}, M:{self.network[layer_index]["M"]}')

        feasible_configuration = False
        if any([x['M'] > self.M for x in self.network.values()]):
            return feasible_configuration
        else:
            feasible_configuration = True
            return feasible_configuration


    

    def update_network(self, target_layer_index=None):
        latency = {}
        weight_area = {}
        max_latency = [0,0]
        max_area = [0,0]
        for layer_index, layer in self.network.items():
            latency[layer_index] = layer['OX'] * layer['OY'] * layer['Ct'] * layer['FXt'] * layer['FYt'] * layer['Kt']
            weight_area[layer_index] = layer['C'] * layer['K'] * layer['FX'] * layer['FY']
            if latency[layer_index] > max_latency[0]:
                max_area[0] = 0
                max_latency[0] = latency[layer_index]
                max_latency[1] = layer_index
                if weight_area[layer_index] > max_area[0]:
                    max_area[0] = weight_area[layer_index]
                    max_area[1] = layer_index
            if latency[layer_index] == max_latency[0]:
                if weight_area[layer_index] > max_area[0]:
                    max_area[0] = weight_area[layer_index]
                    max_area[1] = layer_index

        if target_layer_index == None:
            target_layer_index = max_area[1]
        k_pf = [('K',x) for x in prime_factors(self.network[target_layer_index]['K'])]
        if k_pf != []:
            pf = k_pf
            pf.sort(key = lambda x: x[1])
            pf_cut = pf[0]
            self.network[target_layer_index][pf_cut[0]] /= pf_cut[1]
            self.network[target_layer_index][f'{pf_cut[0]}t'] *= pf_cut[1]
            self.network[target_layer_index][f'M'] *= pf_cut[1]
            logger.info(f'Network update: Layer {target_layer_index} cut {pf_cut[0]} by {pf_cut[1]} --> {pf_cut[0]}: {self.network[target_layer_index][pf_cut[0]]} {pf_cut[0]}t: {self.network[target_layer_index][pf_cut[0]+"t"]}, M:{self.network[target_layer_index]["M"]}')
        else:
            c_pf = [('C',x) for x in prime_factors(self.network[target_layer_index]['C'])]
            fx_pf = [('FX',x) for x in prime_factors(self.network[target_layer_index]['FX'])]
            fy_pf = [('FY',x) for x in prime_factors(self.network[target_layer_index]['FY'])]
            
            pf = c_pf + fx_pf + fy_pf  
            pf.sort(key = lambda x: x[1])
            pf_cut = pf[0]
            self.network[target_layer_index][pf_cut[0]] /= pf_cut[1]
            self.network[target_layer_index][f'{pf_cut[0]}t'] *= pf_cut[1]
            self.network[target_layer_index][f'M'] *= pf_cut[1]
            logger.info(f'Network update: Layer {target_layer_index} cut {pf_cut[0]} by {pf_cut[1]} --> {pf_cut[0]}: {self.network[target_layer_index][pf_cut[0]]} {pf_cut[0]}t: {self.network[target_layer_index][pf_cut[0]+"t"]}, M:{self.network[target_layer_index]["M"]}')

        feasible_configuration = False
        if any([x['M'] > self.M for x in self.network.values()]):
            return feasible_configuration
        else:
            feasible_configuration = True
            return feasible_configuration
