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
    def __init__(self, D1, D2, network, D3, M, ox_unrolling_scheme, verbose=0):
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.M = M
        self.network = network
        self.item_pool = None
        self.ox_unrolling_scheme = ox_unrolling_scheme
        self.verbose = verbose

    def generate(self):
        items = []
        feasible_tile_configuration = True
        for ii_n, n in self.network.items():
            k_pf = [('K',x) for x in prime_factors(n['K'])]
            gm_pf = [('Gm',x) for x in prime_factors(n['Gm'])]
            c_pf = [('C',x) for x in prime_factors(n['C'])]
            fx_pf = [('FX',x) for x in prime_factors(n['FX'])]
            fy_pf = [('FY',x) for x in prime_factors(n['FY'])]
            
            d1_comb = []
            d2_comb = []
            max_d1, max_d2 = 0,0

            for k in range(len(k_pf) + len(gm_pf) +1):
                for c in itertools.combinations(k_pf + gm_pf,k):
                    if np.prod([x[1] for x in c]) <= self.D1 and np.prod([x[1] for x in c]) > max_d1:
                        max_d1 = np.prod([x[1] for x in c])
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
                        max_d2 = np.prod([x[1] for x in c])
                        cx = []
                        for lpf in c:
                            if lpf[0] not in [x[0] for x in cx]:
                                cx.append(list(lpf))
                            else:
                                lpfx = next((x for x in cx if x[0] == lpf[0]),None)
                                lpfx[1] *= lpf[1]
                        cx = tuple([tuple(x) for x in cx])
                        d2_comb = cx

            OXu = next((x for x in self.ox_unrolling_scheme if x[0] == ii_n),[1,1])[1]
            
            d1_combx = tuple([x for x in d1_comb if x[0] != 'Gm'])
            item_repetition = np.prod([x for k,x in n.items() if k in ['G','K','FX','FY','C']]) / np.prod([x[1] for x in d1_combx + d2_comb]) * OXu
            if item_repetition > self.D3:
                feasible_tile_configuration = False
                return None, feasible_tile_configuration, (n['layer_id'], d1_comb, d2_comb, OXu)

            width = np.prod([x[1] for x in d1_comb])
            depth = np.prod([x[1] for x in d2_comb])

            d3_comb = []
            for loop_type in ['K','C','FX','FY']:
                if loop_type in ['C','FX','FY']:
                    comb = d2_comb
                else:
                    comb = d1_comb
                lp = next((x for x in comb if x[0] == loop_type),(loop_type,1))
                if lp != None:
                    if int(n[loop_type] / lp[1]) != 1:
                        d3_comb.append((loop_type, int(n[loop_type] / lp[1])))
            d3_comb.append(('OX',OXu))
            d3_comb.append(('G',n['G']))
#            self.network[ii_n]['OXt'] = int(self.network[ii_n]['OX'] / OXu)
#            self.network[ii_n]['OX'] = OXu
            if d1_combx == tuple():
                d1_comb = (('K',1),)
            if d2_comb == tuple():
                d2_comb = (('C',1),)
            items.append(Item(width=int(width), depth=int(depth), height=int(n['M']), layer_index=ii_n, tile_index=int(item_repetition), \
                    D1_unroll = tuple([x for x in d1_comb if x[0] != 'Gm']), D2_unroll = tuple(d2_comb), D3_unroll = tuple(d3_comb)))
            if self.verbose == 2:
                logger.info(f"Generated #{len(items):4} {items[-1]}")

        self.item_pool = set(items)
        if self.verbose == 1:
            logger.info(f"Generated Items #{len(items):4}")
        return set(items), feasible_tile_configuration, None


    def set_init_M(self, ox_unrolling_scheme):

        
        # Make sure that C, FX, FY, G fit in D2 * D3
        for layer_index, layer in self.network.items():
            fitting = False
            if layer_index in [x[0] for x in ox_unrolling_scheme]:
                oxu = next((x for x in ox_unrolling_scheme if x[0] == layer_index),None)
                self.network[layer_index]['OXt'] = self.network[layer_index]['OXt'] / oxu[1]
                self.network[layer_index]['OX'] = oxu[1]


            while not fitting:
                if np.prod([layer['FX'] * layer['FY'] * layer['C']]) <= self.D2 * self.D3:
                    fitting = True
                    continue
                c_pf = [('C',x) for x in prime_factors(layer['C'])]
                fx_pf = [('FX',x) for x in prime_factors(layer['FX'])]
                fy_pf = [('FY',x) for x in prime_factors(layer['FY'])]
                
                pf = c_pf + fx_pf + fy_pf  
                pf.sort(key = lambda x: x[1])
                pf.reverse()
                pf_cut = pf[0]
                self.network[layer_index][pf_cut[0]] /= pf_cut[1]
                self.network[layer_index][f'{pf_cut[0]}t'] *= pf_cut[1]
                self.network[layer_index][f'M'] *= pf_cut[1]
                if self.verbose == 2:
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
                if self.verbose == 2:
                    logger.info(f'Mapping init: Layer {layer_index} cut {pf_cut[0]} by {pf_cut[1]} --> {pf_cut[0]}: {self.network[layer_index][pf_cut[0]]} {pf_cut[0]}t: {self.network[layer_index][pf_cut[0]+"t"]}, M:{self.network[layer_index]["M"]}')

        for layer_index, layer in self.network.items():
            fitting = False
            while not fitting:
                if np.prod([layer['G']]) <= self.D3:
                    fitting = True
                    continue
                g_pf = [('G',x) for x in prime_factors(layer['G'])]
                pf = g_pf
                pf.sort(key = lambda x: x[1])
                pf_cut = pf[0]
                self.network[layer_index][pf_cut[0]] /= pf_cut[1]
                self.network[layer_index][f'{pf_cut[0]}t'] *= pf_cut[1]
                self.network[layer_index][f'M'] *= pf_cut[1]
                if self.verbose == 2:
                    logger.info(f'Mapping init: Layer {layer_index} cut {pf_cut[0]} by {pf_cut[1]} --> {pf_cut[0]}: {self.network[layer_index][pf_cut[0]]} {pf_cut[0]}t: {self.network[layer_index][pf_cut[0]+"t"]}, M:{self.network[layer_index]["M"]}')

            fitting = True
            self.network[layer_index]['Gm'] = 1
            g_pf = None
            while fitting:
                g_pf = [('G',x) for x in prime_factors(layer['Gt'])]
                pf = g_pf
                pf.sort(key = lambda x: x[1])
                if pf == []:
                    break
                pf_cut = pf[0]
                if self.network[layer_index]['Gm']*pf_cut[1] > self.D1:
                    fitting = False
                    break

                self.network[layer_index]['Gt'] /= pf_cut[1]
                self.network[layer_index]['Gm'] *= pf_cut[1]
                self.network[layer_index][f'M'] /= pf_cut[1]
                if self.verbose == 2:
                    logger.info(f'Mapping init: Layer {layer_index} cut {pf_cut[0]}t by {pf_cut[1]} --> {pf_cut[0]}t: {self.network[layer_index][pf_cut[0]+"t"]} {pf_cut[0]}m: {self.network[layer_index][pf_cut[0]+"m"]}, M:{self.network[layer_index]["M"]}')




        feasible_configuration = False
        if any([x['M'] > self.M for x in self.network.values()]):
            breakpoint()
            return feasible_configuration
        else:
            feasible_configuration = True
            return feasible_configuration


    def update_mapping2(self, target_layer_index_r=None):
        d1_comb = target_layer_index_r[1]
        d1_comb_extended, d2_comb_extended = [], []

        for c in d1_comb:
            c_pf = prime_factors(c[1])
            for c_prime_factor in c_pf:
                d1_comb_extended.append((c[0],int(c_prime_factor)))
        d2_comb = target_layer_index_r[2]
        for c in d2_comb:
            c_pf = prime_factors(c[1])
            for c_prime_factor in c_pf:
                d2_comb_extended.append((c[0],int(c_prime_factor)))
        oxu = target_layer_index_r[3]
        target_layer_index_r = target_layer_index_r[0]

        feasible_configuration = False

        network_copy = copy.deepcopy(self.network)
        target_layer_index = next((k for k,v in self.network.items() if v['layer_id'] == target_layer_index_r),None)
        k_pf = [('K',int(x)) for x in prime_factors(self.network[target_layer_index]['K'])]
        g_pf = [('G',int(x)) for x in prime_factors(self.network[target_layer_index]['G'])]
        c_pf = [('C',int(x)) for x in prime_factors(self.network[target_layer_index]['C'])]
        fx_pf = [('FX',int(x)) for x in prime_factors(self.network[target_layer_index]['FX'])]
        fy_pf = [('FY',int(x)) for x in prime_factors(self.network[target_layer_index]['FY'])]

        # Find best D3 comb, including OXu in the unrollings
        # Prioritize C, FX, FY loops

        # clean c_pf, fx_pf, fy_pf based on D1_comb
        for d2_cpf in d2_comb_extended:
            if d2_cpf[0] == 'C':
                c_pf.remove(('C',d2_cpf[1]))
            if d2_cpf[0] == 'FX':
                fx_pf.remove(('FX',d2_cpf[1]))
            if d2_cpf[0] == 'FY':
                fy_pf.remove(('FY',d2_cpf[1]))
        for d1_cpf in d1_comb_extended:
            k_pf.remove(('K',d1_cpf[1]))
        oxu_loop = ('OX',oxu)
        max_utilization = 0
        d3_comb = [oxu_loop]
        for k in range(len(c_pf + fx_pf + fy_pf)+1):
            for comb in itertools.combinations(c_pf + fx_pf + fy_pf,k):
                if oxu * np.prod([c[1] for c in comb]) <= self.D3 and oxu * np.prod([c[1] for c in comb]) > max_utilization:
                    max_utilization = oxu * np.prod([c[1] for c in comb])
                    cx = [oxu_loop] + [tuple([x[0],int(x[1])]) for x in comb]
                    d3_comb = cx
        d3_len = np.prod([c[1] for c in d3_comb])
        d3_comb_new = copy.deepcopy(d3_comb)
        for k in range(len(k_pf + g_pf)+1):
            for comb in itertools.combinations(k_pf + g_pf, k):
                if d3_len * np.prod([c[1] for c in comb]) <= self.D3 and d3_len * np.prod([c[1] for c in comb]) > max_utilization:
                    max_utilization = d3_len * np.prod([c[1] for c in comb])
                    cx = d3_comb + [tuple([x[0],int(x[1])]) for x in comb]
                    d3_comb_new = cx

        d3_comb = d3_comb_new
        for d3_cpf in d3_comb:
            if d3_cpf[0] == 'C':
                c_pf.remove(('C',d3_cpf[1]))
            if d3_cpf[0] == 'FX':
                fx_pf.remove(('FX',d3_cpf[1]))
            if d3_cpf[0] == 'FY':
                fy_pf.remove(('FY',d3_cpf[1]))
            if d3_cpf[0] == 'K':
                k_pf.remove(('K',d3_cpf[1]))
            if d3_cpf[0] == 'G':
                g_pf.remove(('G',d3_cpf[1]))


        for pf in c_pf + fx_pf + fy_pf + k_pf + g_pf:
            network_copy[target_layer_index][pf[0]] /= pf[1]
            network_copy[target_layer_index][f'{pf[0]}t'] *= pf[1]
            network_copy[target_layer_index][f'M'] *= pf[1]

        if any([x['M'] > self.M for x in network_copy.values()]):
            return feasible_configuration, None
        else:
            self.network = network_copy
            feasible_configuration = True
            return feasible_configuration, (d1_comb, d2_comb, d3_comb)
        return feasible_configuration   

    def update_mapping(self, target_layer_index_r=None):
        if target_layer_index_r != None:
            d1_comb = target_layer_index_r[1]
            d2_comb = target_layer_index_r[2]
            target_layer_index_r = target_layer_index_r[0]

        latency = []
        weight_area = []
        vals = []
        for layer_index, layer in self.network.items():
            latency.append(layer['OXt'] * layer['OY'] * layer['Ct'] * layer['FXt'] * layer['FYt'] * layer['Kt'] * layer['Gt'])
            weight_area.append(layer['C'] * layer['K'] * layer['FX'] * layer['FY'] * layer['OX'] * layer['G'])
            vals.append({'network_index':layer['layer_id'],'latency':latency[-1],'weight_area':weight_area[-1]})
        df = pd.DataFrame(vals)
        df = df.sort_values(by=['latency','weight_area'],ascending=[True,False],ignore_index=True)
#        df = df.sort_values(by=['weight_area'],ascending=[False],ignore_index=True)
        feasible_configuration = False

        for i,r in df.iterrows():
            target_layer_index = r.network_index
            if target_layer_index_r != None:
                if target_layer_index != target_layer_index_r:
                    continue
            network_copy = copy.deepcopy(self.network)
            target_layer_index = next((k for k,v in self.network.items() if v['layer_id'] == target_layer_index),None)
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
                if self.verbose == 2:
                    logger.info(f'Mapping update: Layer {target_layer_index} cut {pf_cut[0]} by {pf_cut[1]} --> {pf_cut[0]}: {network_copy[target_layer_index][pf_cut[0]]} {pf_cut[0]}t: {network_copy[target_layer_index][pf_cut[0]+"t"]}, M:{network_copy[target_layer_index]["M"]}')
            else:
                pf = c_pf + fx_pf + fy_pf  
                pf.sort(key = lambda x: x[1])
                pf.reverse()
                pf_cut = pf[0]
                network_copy[target_layer_index][pf_cut[0]] /= pf_cut[1]
                network_copy[target_layer_index][f'{pf_cut[0]}t'] *= pf_cut[1]
                network_copy[target_layer_index][f'M'] *= pf_cut[1]
                if self.verbose == 2:
                    logger.info(f'Network update: Layer {target_layer_index} cut {pf_cut[0]} by {pf_cut[1]} --> {pf_cut[0]}: {network_copy[target_layer_index][pf_cut[0]]} {pf_cut[0]}t: {network_copy[target_layer_index][pf_cut[0]+"t"]}, M:{network_copy[target_layer_index]["M"]}')

            if any([x['M'] > self.M for x in network_copy.values()]):
                logger.info('No modify')
                continue
            else:
                self.network = network_copy
                feasible_configuration = True
                return feasible_configuration
        return feasible_configuration
