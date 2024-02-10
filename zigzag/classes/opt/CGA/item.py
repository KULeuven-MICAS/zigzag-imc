import numpy as np
import itertools
from zigzag.classes.opt.CGA.utils import prime_factors
from loguru import logger
import copy
import pandas as pd

class Item():
    # D1 = width
    # D2 = depth
    def __init__(self, *, height, width, depth, layer_index, tile_index,D1_unroll, D2_unroll, D3_unroll):
        self.height = height
        self.depth = depth
        self.width = width
        self.layer_index = layer_index
        self.tile_index = tile_index
        self.volume = height * width * depth
        self.area = width * depth
        self.D1_unroll = D1_unroll 
        self.D2_unroll = D2_unroll 
        self.D3_unroll = D3_unroll 
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
            width = np.prod([x[1] for x in d1_comb])
            depth = np.prod([x[1] for x in d2_comb])
            if int(item_repetition) == 0:
                breakpoint()

            d3_comb = []
            for loop_type in ['K','C','FX','FY']:
                if loop_type in ['C','FX','FY']:
                    comb = d2_comb
                else:
                    comb = d1_comb
                lp = next((x for x in comb if x[0] == loop_type),None)
                if lp != None:
                    d3_comb.append((loop_type, n[loop_type] / lp[1]))
            if d1_comb == tuple():
                d1_comb = (('K',1),)
            if d2_comb == tuple():
                d2_comb = (('C',1),)
            items.append(Item(width=int(width), depth=int(depth), height=int(n['M']), layer_index=ii_n, tile_index=int(item_repetition), \
                    D1_unroll = tuple(d1_comb), D2_unroll = tuple(d2_comb), D3_unroll = tuple(d3_comb)))
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
                logger.info(f'Mapping update: Layer {layer_index} cut {pf_cut[0]} by {pf_cut[1]} --> {pf_cut[0]}: {self.network[layer_index][pf_cut[0]]} {pf_cut[0]}t: {self.network[layer_index][pf_cut[0]+"t"]}, M:{self.network[layer_index]["M"]}')

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
                logger.info(f'Mapping init: Layer {layer_index} cut {pf_cut[0]} by {pf_cut[1]} --> {pf_cut[0]}: {self.network[layer_index][pf_cut[0]]} {pf_cut[0]}t: {self.network[layer_index][pf_cut[0]+"t"]}, M:{self.network[layer_index]["M"]}')

        feasible_configuration = False
        if any([x['M'] > self.M for x in self.network.values()]):
            return feasible_configuration
        else:
            feasible_configuration = True
            return feasible_configuration


    

    def update_mapping(self, target_layer_index=None):
        latency = []
        weight_area = []
        vals = []
        for layer_index, layer in self.network.items():
            latency.append(layer['OX'] * layer['OY'] * layer['Ct'] * layer['FXt'] * layer['FYt'] * layer['Kt'])
            weight_area.append(layer['C'] * layer['K'] * layer['FX'] * layer['FY'])
            vals.append({'network_index':layer_index,'latency':latency[-1],'weight_area':weight_area[-1]})
        df = pd.DataFrame(vals)
        df = df.sort_values(by=['latency','weight_area'],ascending=[True,False],ignore_index=True)
        feasible_configuration = False
        for i,r in df.iterrows():
            target_layer_index = r.network_index
            network_copy = copy.deepcopy(self.network)
            k_pf = [('K',x) for x in prime_factors(self.network[target_layer_index]['K'])]
            c_pf = [('C',x) for x in prime_factors(self.network[target_layer_index]['C'])]
            fx_pf = [('FX',x) for x in prime_factors(self.network[target_layer_index]['FX'])]
            fy_pf = [('FY',x) for x in prime_factors(self.network[target_layer_index]['FY'])]
            if all([x == [] for x in [k_pf, c_pf, fx_pf, fy_pf]]):
                continue
            if k_pf != []:
                pf = k_pf
                pf.sort(key = lambda x: x[1])
                pf_cut = pf[0]
                network_copy[target_layer_index][pf_cut[0]] /= pf_cut[1]
                network_copy[target_layer_index][f'{pf_cut[0]}t'] *= pf_cut[1]
                network_copy[target_layer_index][f'M'] *= pf_cut[1]
                logger.info(f'Mapping update: Layer {target_layer_index} cut {pf_cut[0]} by {pf_cut[1]} --> {pf_cut[0]}: {network_copy[target_layer_index][pf_cut[0]]} {pf_cut[0]}t: {network_copy[target_layer_index][pf_cut[0]+"t"]}, M:{network_copy[target_layer_index]["M"]}')
            else:
                pf = c_pf + fx_pf + fy_pf  
                pf.sort(key = lambda x: x[1])
                pf_cut = pf[0]
                network_copy[target_layer_index][pf_cut[0]] /= pf_cut[1]
                network_copy[target_layer_index][f'{pf_cut[0]}t'] *= pf_cut[1]
                network_copy[target_layer_index][f'M'] *= pf_cut[1]
                logger.info(f'Network update: Layer {target_layer_index} cut {pf_cut[0]} by {pf_cut[1]} --> {pf_cut[0]}: {network_copy[target_layer_index][pf_cut[0]]} {pf_cut[0]}t: {network_copy[target_layer_index][pf_cut[0]+"t"]}, M:{network_copy[target_layer_index]["M"]}')

            if any([x['M'] > self.M for x in network_copy.values()]):
                continue
            else:
                self.network = network_copy
                feasible_configuration = True
                return feasible_configuration

        return feasible_configuration
