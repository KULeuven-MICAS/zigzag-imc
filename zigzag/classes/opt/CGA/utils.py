import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def plot_item_allocation(layer_list, bin_dict, D3, height, D1, D2):
    r_val, c_val = 1, 1
    pf = prime_factors(D3)
    for ii_pf, pf in enumerate(pf):
        if ii_pf % 2 == 0:
            r_val *= pf
        else:
            c_val *= pf

   
    specs = [[{'type': 'surface'}] * c_val] * r_val

    fig = make_subplots(rows=r_val, cols=c_val, specs=specs)

    color= px.colors.qualitative.Vivid

    for d3 in range(D3):
        z_offset = 0
        bin_macro = bin_dict[d3]
        layer_list_tmp = [x for x in layer_list if x.id in bin_macro]
        row=int(np.floor(d3/c_val))+1
        col= int(d3 % c_val) + 1
        for l in layer_list_tmp:
            for si in l.superitem_set:
                si_z_offset = 0
                for i in si.item_set:
                    x = np.ones(8,dtype=np.int64)
                    y = np.ones(8,dtype=np.int64)
                    z = np.ones(8,dtype=np.int64) * (z_offset + si_z_offset)
                    x[0] = i.x_pos
                    x[1] = i.x_pos 
                    x[2] = i.x_pos + i.width
                    x[3] = i.x_pos + i.width 
                    x[4:] = x[:4]
                    y[0] = i.y_pos
                    y[1] = i.y_pos + i.depth
                    y[2] = i.y_pos + i.depth
                    y[3] = i.y_pos
                    y[4:] = y[:4]
                    z[4:] += i.height 
                    text = f'D1:{i.width} D2:{i.depth} M:{i.height}'
                    fig.append_trace( 
                        go.Mesh3d(
                            # 8 vertices of a cube
                            x=x, y=y, z=z,
                            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                            opacity=0.6,
                            color=color[i.layer_index],
                            flatshading = True,
                            name=f'L {i.layer_index} Tile {i.tile_index}',
                            text=text,
                            legendgroup=f'Layer {i.layer_index}'
                        ), row=int(d3 // c_val)+1,col=int(np.floor(d3 % c_val))+1)
                    si_z_offset += i.height
            z_offset += l.height
        if d3 == 0:
            fig['layout']['scene'][f'xaxis'].update(title='D1', range=[0, D1])
            fig['layout']['scene'][f'yaxis'].update(title='D2', range=[0, D2])
            fig['layout']['scene'][f'zaxis'].update(title='M', range=[0, height])
        else:
            fig['layout'][f'scene{d3+1}'][f'xaxis'].update(title='D1', range=[0, D1])
            fig['layout'][f'scene{d3+1}'][f'yaxis'].update(title='D2', range=[0, D2])
            fig['layout'][f'scene{d3+1}'][f'zaxis'].update(title='M', range=[0, height])



    fig.update_traces(showlegend=True, selector=dict(type='mesh3d'))
    fig.show()


vgg_16_network = {
        0: {'K':32, 'C':3,    'FX':3,'FY':3, 'OX':1, 'OYt':112, 'Ct':1, 'OXt':112, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':1},
        1: {'K':64, 'C':32,   'FX':3,'FY':3, 'OX':1, 'OYt':112, 'Ct':1, 'OXt':112, 'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':2},
        2: {'K':64, 'C':64,   'FX':3,'FY':3, 'OX':1, 'OYt':56,  'Ct':1, 'OXt':56,  'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':3},
        3: {'K':112, 'C':64,  'FX':3,'FY':3, 'OX':1, 'OYt':56,  'Ct':1, 'OXt':56,  'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':4},
        4: {'K':112, 'C':112, 'FX':3,'FY':3, 'OX':1, 'OYt':28,  'Ct':1, 'OXt':28,  'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':5},
        5: {'K':112, 'C':112, 'FX':3,'FY':3, 'OX':1, 'OYt':28,  'Ct':1, 'OXt':28,  'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':6},
        6: {'K':224, 'C':112, 'FX':3,'FY':3, 'OX':1, 'OYt':28,  'Ct':1, 'OXt':28,  'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':7},
        7: {'K':224, 'C':224, 'FX':3,'FY':3, 'OX':1, 'OYt':14,  'Ct':1, 'OXt':14,  'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':8},
        8: {'K':224, 'C':224, 'FX':3,'FY':3, 'OX':1, 'OYt':14,  'Ct':1, 'OXt':14,  'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':9},
        9: {'K':224, 'C':224, 'FX':3,'FY':3, 'OX':1, 'OYt':14,  'Ct':1, 'OXt':14,  'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':10},
       10: {'K':224, 'C':224, 'FX':3,'FY':3, 'OX':1, 'OYt':7,   'Ct':1, 'OXt':7,   'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':11},
       11: {'K':224, 'C':224, 'FX':3,'FY':3, 'OX':1, 'OYt':7,   'Ct':1, 'OXt':7,   'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':12},
       12: {'K':224, 'C':224, 'FX':3,'FY':3, 'OX':1, 'OYt':7,   'Ct':1, 'OXt':7,   'FXt':1, 'FYt':1, 'Kt':1, 'M': 1, 'layer_id':13}
}
vgg_16_network_validation = {
        0: {'K':1, 'C':3,    'FX':3,'FY':3, 'OX':1,  'OYt':112, 'Ct':1,  'OXt':112, 'FXt':1, 'FYt':1, 'Kt':32, 'M': 32, 'layer_id':1},
        1: {'K':2, 'C':32,   'FX':3,'FY':3, 'OX':1,  'OYt':112, 'Ct':1,  'OXt':112, 'FXt':1, 'FYt':1, 'Kt':32, 'M': 32, 'layer_id':2},
        2: {'K':2, 'C':64,   'FX':3,'FY':3, 'OX':1,  'OYt':56,  'Ct':1,  'OXt':56, 'FXt':1, 'FYt':1, 'Kt':32, 'M': 32, 'layer_id':3},
        3: {'K':4, 'C':64,   'FX':3,'FY':3, 'OX':1,  'OYt':56,  'Ct':1,  'OXt':56, 'FXt':1, 'FYt':1,  'Kt':28, 'M': 28, 'layer_id':4},
        4: {'K':4, 'C':112,  'FX':3,'FY':3, 'OX':1,  'OYt':28,  'Ct':1,  'OXt':28, 'FXt':1, 'FYt':1,  'Kt':28, 'M': 28, 'layer_id':5},
        5: {'K':4, 'C':112,  'FX':3,'FY':3, 'OX':1,  'OYt':28,  'Ct':1,  'OXt':28, 'FXt':1, 'FYt':1,  'Kt':28, 'M': 28, 'layer_id':6},
        6: {'K':8, 'C':112,  'FX':3,'FY':3, 'OX':1,  'OYt':28,  'Ct':1,  'OXt':28, 'FXt':1, 'FYt':1,  'Kt':28, 'M': 28, 'layer_id':7},
        7: {'K':8, 'C':224,  'FX':3,'FY':3, 'OX':1,  'OYt':14,  'Ct':1,  'OXt':14, 'FXt':1, 'FYt':1,  'Kt':32, 'M': 32, 'layer_id':8},
        8: {'K':8, 'C':224,  'FX':3,'FY':3, 'OX':1,  'OYt':14,  'Ct':1,  'OXt':14, 'FXt':1, 'FYt':1,  'Kt':32, 'M': 32, 'layer_id':9},
        9: {'K':8, 'C':224,  'FX':3,'FY':3, 'OX':1,  'OYt':14,  'Ct':1,  'OXt':14, 'FXt':1, 'FYt':1,  'Kt':32, 'M': 32, 'layer_id':10},
       10: {'K':8, 'C':224,  'FX':3,'FY':3, 'OX':1,  'OYt':7,   'Ct':1,  'OXt':7,  'FXt':1, 'FYt':1,  'Kt':32, 'M': 32, 'layer_id':11},
       11: {'K':8, 'C':224,  'FX':3,'FY':3, 'OX':1,  'OYt':7,   'Ct':1,  'OXt':7,  'FXt':1, 'FYt':1,  'Kt':32, 'M': 32, 'layer_id':12},
       12: {'K':8, 'C':224,  'FX':3,'FY':3, 'OX':1,  'OYt':7,   'Ct':1,  'OXt':7,  'FXt':1, 'FYt':1,  'Kt':32, 'M': 32, 'layer_id':13}
}


if __name__ == "__main__":
    plot_item_allocation([], None, 1, 10)


