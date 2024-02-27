import numpy as np
import zigzag.classes.opt.CGA.utils as prime_factors


def get_latency(network, D1, D2, D3, M):


if __name__ == "__main__":
    # LAYERS MUST START FROM INDEX ZERO!
    network = {0:{'K':16,  'C':3,   'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               1:{'K':16,  'C':16,  'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               2:{'K':16,  'C':16,  'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               3:{'K':32,  'C':16,  'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               4:{'K':32,  'C':32,  'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               5:{'K':32,  'C':16,  'FX':1,'FY':1, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               6:{'K':64,  'C':32,  'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               7:{'K':64,  'C':64,  'FX':3,'FY':3, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               8:{'K':64,  'C':32,  'FX':1,'FY':1, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1},
               9:{'K':10,  'C':64,  'FX':1,'FY':1, 'OX':1, 'OY':1, 'Ct':1, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1}}

    D1 = 1
    D2 = 24
    D3 = 384
    M = 256


