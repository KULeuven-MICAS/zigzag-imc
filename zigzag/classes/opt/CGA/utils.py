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

if __name__ == "__main__":
    plot_item_allocation([], None, 1, 10)


