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
            df_data.append({'layer':l.layer.id,
                'loop_dim_size':l.layer.loop_dim_size,
                'area':l.imc_area,
                'latency_cc':l.ideal_temporal_cycle,
                'weight_loading_cc':l.SS_comb,
                'tclk':l.tclk,
                'tclk_breakdown':l.tclk_breakdown,
                'mem_energy_breakdown':l.mem_energy_breakdown,
                'MAC_energy_breakdown':l.MAC_energy_breakdown,
                'energy_total': sum([sum(v) for v in l.mem_energy_breakdown.values()] + [x for x in l.MAC_energy_breakdown.values()]),
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
    df = get_df("./outputs_resnet8_9x9/")
    df = df.sort_values(by=['M','cfg'],ignore_index=True)
    dfa = df.drop_duplicates('layer')
    weights = 0
    for i,r in dfa.iterrows():
        lds = r.loop_dim_size
        try:
            weights += lds['K'] * lds['C'] * lds['FX'] * lds['FY']
        except:
            weights += lds['K'] * lds['C']

    print("Weights: ",weights)
    df['latency'] = df['latency_cc'] * df['tclk']
    df['latency_total'] = df['latency_cc'] + df['weight_loading_cc']
    df['ActD1'] = df['D1'] * df['M']
    df['EDP']  = df['energy_total'] * df['latency_total']
    df['Dtot'] = df['D1'] * df['D2']
    df_fixed_array_size = df[(df.ActD1 == 128*9) & (df.D2 ==128*9) & (df.layer == 2)]
    df = df.groupby(['cfg'], sort=False, as_index=False).agg({'bw': ['max'],'bw_weight':['max'],'energy_total':['sum'], 'latency_total':['sum'],'tclk':['mean'],'area':['mean'],'M':['mean'],'D2':['mean']})
    df.columns = [x[0] for x in df.columns]
    df['M'] = df['M'].astype(str)
    #dfx = df.sort_values(by=['energy_total','latency_total'],ascending=[True,True],ignore_index=True).drop_duplicates('cfg')
    dfx = df
    dfx = dfx.sort_values(by=['bw'],ignore_index=True)
    dfx['bw'] = dfx['bw'].astype(str)
    df_fixed_array_size['M'] = df_fixed_array_size['M'].astype(str)
    fig = px.scatter(dfx,'area','latency_total',color='bw',hover_data=['cfg'],log_x=True,log_y=True)
    fig.show()
    fig = px.scatter(dfx,'area','energy_total',color='M',hover_data=['cfg'],log_x=True,log_y=True)
    fig.show()
    fig = px.scatter(dfx,'bw_weight','latency_total',color='M',hover_data=['cfg'],log_x=False,log_y=True)
    fig.show()
    breakpoint()
    # stupid plot to show bandwidth, area, latency tradeoff
    fig = px.scatter(df_fixed_array_size, 'latency_cc', 'area', size='bw',color='M', text='bw',hover_data=['cfg'], log_x=True, log_y=True)
    fig.update_traces(textposition='top center')
    fig.show()

    # tclk breakdown
    df_data = []
    for i,r in df_fixed_array_size.iterrows():
        dd = r.tclk_breakdown
        for k,v in dd.items():
            val = {}
            val['cfg'] = r.cfg
            val['latency'] = v
            val['source'] = k
            val['area'] = r.area
            df_data.append(val)
    df_tclk = pd.DataFrame(df_data)
    fig = px.bar(df_tclk, 'cfg','latency',color='source')
    fig.show()

    # energy breakdown
    df_data = []
    for i,r in df_fixed_array_size.iterrows():
        dd = r.mem_energy_breakdown
        for k,v in dd.items():
            val = {}
            val['cfg'] = r.cfg
            val['energy'] = sum(v)
            val['source'] = k
            val['area'] = r.area
            df_data.append(val)
        dd = r.MAC_energy_breakdown
        for k,v in dd.items():
            val = {}
            val['cfg'] = r.cfg
            val['energy'] = v
            val['source'] = k
            val['area'] = r.area
            df_data.append(val)

    df_energy = pd.DataFrame(df_data)
    fig = px.bar(df_energy, 'cfg','energy',color='source')
    fig.show()
    breakpoint()


if __name__ == "__main__":
    fig_plot()
