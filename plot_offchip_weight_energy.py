import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import pandas as pd
import pickle
import os
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import plotly.express as px
import shutil

import zigzag

def get_df(directory):
    power_values, area_values = [], []
    df = pd.DataFrame()
    ii  = 0
    for filename in os.listdir(directory):
        ii+=1
        if filename.split(".")[-1] != 'pkl':
            continue
        f = os.path.join(directory, filename)
        
        with open(f, 'rb') as infile:
            data = pickle.load(infile)

        df_data = []
        for l in data:
            all_weight_unrolled = False
            weight_tm = [j for i in l.temporal_mapping.mapping_dic_origin['W'] for j in i]
            if all([x[0] not in ['K','C','FX','FY'] for x in weight_tm]):
                all_weight_unrolled = True
            df_data.append({'layer':l.layer.id,
                'loop_dim_size':l.layer.loop_dim_size,
                'area':l.imc_area,
                'latency_cc':l.ideal_temporal_cycle,
                'weight_loading_cc':l.SS_comb,
                'dram_loading_SS_per_period':l.port_activity_collect[5]['r_port_1'][0].SS_per_period,
                'dram_loading_SS':l.port_activity_collect[5]['r_port_1'][0].SS,
                'weight_tm':l.temporal_mapping.mapping_dic_origin['W'],
                'weight_unrolled':all_weight_unrolled,
                'tclk':l.tclk,
                'tclk_breakdown':l.tclk_breakdown,
                'mem_energy_breakdown':l.mem_energy_breakdown,
                'MAC_energy_breakdown':l.MAC_energy_breakdown,
                'energy_total': sum([sum(l.mem_energy_breakdown['I'][:-1]), sum(l.mem_energy_breakdown['O'][:-1]), sum(l.mem_energy_breakdown['W'])] + [x for x in l.MAC_energy_breakdown.values()]),
                'M': l.accelerator.cores[0].operational_array.unit.group_depth,
                'D1':l.accelerator.cores[0].operational_array.dimensions[0].size,
                'D2':l.accelerator.cores[0].operational_array.dimensions[1].size,
                "cfg": f"M{int(l.accelerator.cores[0].operational_array.unit.group_depth)} D1:{int(l.accelerator.cores[0].operational_array.dimensions[0].size)} D2:{int(l.accelerator.cores[0].operational_array.dimensions[1].size)}",
                "bw": l.mapping_int.unit_mem_data_movement['I'][1].req_mem_bw_aver.rd_out_to_low,
                "bw_weight": l.mapping_int.unit_mem_data_movement['W'][1].req_mem_bw_aver.rd_out_to_low})
        data = pd.DataFrame(df_data)
        if df.empty:
            df = data
        else:
            df = pd.concat([df, data])

    return df

def fig_plot():
    df = get_df("./outputs/")
    df['latency_total'] = df['latency_cc'] + df['weight_loading_cc'] + df['dram_loading_SS_per_period']
    df = df.groupby(['cfg'], sort=False, as_index=False).agg({'bw': ['max'],'bw_weight':['max'],'energy_total':['sum'], 'latency_total':['sum'],'tclk':['mean'],'area':['mean'],'M':['mean'],'D2':['mean']})
    df.columns = [x[0] for x in df.columns]
#    df = df.sort_values(by=['area','latency_total'],ascending=[True,True],ignore_index=True).drop_duplicates(['layer','latency_total'])
    fig = px.scatter(df, 'area', 'energy_total',hover_data=['cfg'])
    fig.show()
    with open("offchip_weights_resnet8_energy.pkl","wb") as infile:
        pickle.dump(df, infile)
    breakpoint()

if __name__ == "__main__":
    fig_plot()
