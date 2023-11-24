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
                'weight_sram_area':l.mem_area_breakdown['weight_sram'],
                'latency_cc':l.ideal_temporal_cycle,
                'weight_loading_cc':l.SS_comb,
                'dram_loading_weight':l.data_loading_cc_per_op['W']['W2_rd_out_to_low'][0],
                'dram_loading_SS':l.port_activity_collect[5]['r_port_1'][0].SS,
                'weight_tm':l.temporal_mapping.mapping_dic_origin['W'],
                'weight_unrolled':all_weight_unrolled,
                'tclk':l.tclk,
                'tclk_breakdown':l.tclk_breakdown,
                'mem_energy_breakdown':l.mem_energy_breakdown,
                'MAC_energy_breakdown':l.MAC_energy_breakdown,
                'M': l.accelerator.cores[0].operational_array.unit.group_depth,
                'D1':l.accelerator.cores[0].operational_array.dimensions[0].size,
                'D2':l.accelerator.cores[0].operational_array.dimensions[1].size,
                'D3':l.accelerator.cores[0].operational_array.dimensions[2].size,
                "cfg": f"M{int(l.accelerator.cores[0].operational_array.unit.group_depth)} D1:{int(l.accelerator.cores[0].operational_array.dimensions[0].size)} D2:{int(l.accelerator.cores[0].operational_array.dimensions[1].size)} D3:{int(l.accelerator.cores[0].operational_array.dimensions[2].size)}",
                "bw": l.mapping_int.unit_mem_data_movement['I'][1].req_mem_bw_aver.rd_out_to_low,
                "bw_weight": l.mapping_int.unit_mem_data_movement['W'][1].req_mem_bw_aver.rd_out_to_low})
        data = pd.DataFrame(df_data)
        if df.empty:
            df = data
        else:
            df = pd.concat([df, data])

    return df

def fig_plot():
    
    network = 'resnet8'
    df = get_df(f"./outputs_{network}/")

    # All weights in IMC macros
    if False:
        df_imc = df.copy(deep=True)
        df_imc = df_imc[df_imc.weight_unrolled == True]
        dfx = df_imc.sort_values(by=['area','latency_cc'],ascending=[True,True],ignore_index=True).drop_duplicates(['layer', 'latency_cc'])
        dfx = dfx.sort_values(by=['layer'],ascending=[True],ignore_index=True)
        cfg_dict = {}
        for layer_id in dfx.layer.unique():
            cfg_dict[layer_id] = []
            for cfg in dfx[dfx.layer == layer_id].cfg.unique():
                cfg_dict[layer_id].append(cfg)
        from itertools import product
        cfg_combinations = list(product(*[v for v in cfg_dict.values()]))
        total_points = []
        for ii_cfgc, cfgc in enumerate(cfg_combinations):
            print(f'{ii_cfgc/len(cfg_combinations):.2f}',end='\r')
            area, latency_cc, energy = 0, 0, 0
            for ii_layer, layer in enumerate(dfx.layer.unique()):
                area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc[ii_layer])].iloc[0]['area']
                latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc[ii_layer])].iloc[0]['latency_cc']
                energy += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc[ii_layer])].iloc[0]['MAC_energy_breakdown'].values()])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc[ii_layer])].iloc[0]['mem_energy_breakdown']['I'][:-1])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc[ii_layer])].iloc[0]['mem_energy_breakdown']['O'][:-1])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc[ii_layer])].iloc[0]['mem_energy_breakdown']['W'][:1])

            total_points.append({'cfg':cfgc,'area':area,'latency_total':latency_cc, 'energy_total':energy})
        df_weights = pd.DataFrame(total_points)

        breakpoint()
        with open(f"imc_weights_{network}.pkl","wb") as infile:
            pickle.dump(df_weights, infile)

    
    # All weights on on-chip SRAM
    df_onchip = df.copy(deep=True)
    dfx = df_onchip
    total_points = []
    for ii_cfgc, cfgc in enumerate(df_onchip.cfg.unique()):
        area, latency_cc, energy = 0, 0, 0
        for ii_layer, layer in enumerate(dfx.layer.unique()):
            area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['area']
            area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_sram_area']
            latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['latency_cc']
            latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_loading_cc']
            energy += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
            energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
            energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
            energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][:-1])

        total_points.append({'cfg':cfgc,'area':area,'latency_total':latency_cc, 'energy_total':energy})
    df_onchip_sram = pd.DataFrame(total_points)

    with open(f"onchip_weights_{network}.pkl","wb") as infile:
        pickle.dump(df_onchip_sram, infile)


    # All weights off-chip
    df_offchip = df.copy(deep=True)
    dfx = df_offchip
    total_points = []
    for ii_cfgc, cfgc in enumerate(df_offchip.cfg.unique()):
        area, latency_cc, energy = 0, 0, 0
        for ii_layer, layer in enumerate(dfx.layer.unique()):
            area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['area']
            latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['latency_cc']
            latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_loading_cc']
            latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['dram_loading_weight']
            latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['dram_loading_SS']
            energy += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
            energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
            energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
            energy += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][0]
            energy += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][-1]
        total_points.append({'cfg':cfgc,'area':area,'latency_total':latency_cc, 'energy_total':energy})
    df_offchip_sram = pd.DataFrame(total_points)

    with open(f"offchip_weights_{network}.pkl","wb") as infile:
        pickle.dump(df_offchip_sram, infile)


if __name__ == "__main__":
    fig_plot()
