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

def get_df(directory, network):
    power_values, area_values = [], []
    df = pd.DataFrame()
    ii  = 0
    for filename in os.listdir(directory):
        ii+=1
        if filename.split(".")[-1] != 'pkl':
            continue
        if filename != f"{network}_all_data_points.pkl":
            continue

        f = os.path.join(directory, filename)
        
        with open(f, 'rb') as infile:
            data = pickle.load(infile)
        df_data = []
        for cc in data:
            for l in cc:
                all_weight_unrolled = False
                weight_tm = [j for i in l.temporal_mapping.mapping_dic_origin['W'] for j in i]
                if all([x[0] not in ['K','C','FX','FY'] for x in weight_tm]):
                    all_weight_unrolled = True
                df_data.append({'layer':l.layer.id,
                    'loop_dim_size':l.layer.loop_dim_size,
                    'area':l.imc_area,
                    'area_db':l.imc_area + l.imc_area_breakdown['cells'],
                    'weight_sram_area':l.mem_area_breakdown['weight_sram'],
                    'act_sram_area':l.mem_area_breakdown['sram_256KB'],
                    'latency_cc':l.ideal_temporal_cycle,
                    'weight_loading_cc':l.SS_comb,
                    'weight_loading_cc_db': max(0, l.SS_comb - (l.port_activity_collect[0]['rw_port_1'][0].period_count - 1) * (l.port_activity_collect[0]['rw_port_1'][0].period)),
                    'dram_loading_weight':l.data_loading_cc_per_op['W']['W2_rd_out_to_low'][0],
                    'dram_loading_SS':l.port_activity_collect[5]['r_port_1'][0].SS,
                    'weight_tm':l.temporal_mapping.mapping_dic_origin['W'],
                    'weight_unrolled':all_weight_unrolled,
                    'tclk':l.tclk,
                    'tclk_breakdown':l.tclk_breakdown,
                    'mem_energy_breakdown':l.mem_energy_breakdown,
                    'MAC_energy_breakdown':l.MAC_energy_breakdown,
                    'MAC_spatial_utilization':l.MAC_spatial_utilization,
                    'total_MAC_count':l.layer.total_MAC_count,
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
    
    network_list = ['mobilenet_v1','ds_cnn','deepautoencoder','resnet8']
    #network_list = ['mobilenet_v1']
    imc_type = 'dimc' 
    for network in network_list:
        print(network)
        print()
        df = get_df(f"./outputs_{network}/{imc_type}/",network)
        # All weights in IMC macros
        if False:#network in ['ds_cnn','resnet8']:
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
            import random

            #cfg_values = random.sample([v for v in cfg_dict.values()],400)
            cfg_settings = []
            cfg_values = [v for v in cfg_dict.values()]
            while len(cfg_settings) != 50000:
                cfg_set = []
                for cf in cfg_values:
                    cfg_set.append(random.choice(cf))
                cfg_set = tuple(cfg_set)
                if cfg_set not in cfg_settings:
                    cfg_settings.append(cfg_set)

            total_points = []
            for ii_cfgc, cfgc in enumerate(cfg_settings):
                print(ii_cfgc, end='\r')
                print(f'{ii_cfgc/50000:.3f}',end='\r')
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

            with open(f"outputs_{network}/{imc_type}/var/imc_weights_{network}.pkl","wb") as infile:
                pickle.dump(df_weights, infile)

        # All weights on on-chip SRAM
 #       df_onchip = df.copy(deep=True)
 #       dfx = df_onchip
 #       total_points = []
 #       for ii_cfgc, cfgc in enumerate(df_onchip.cfg.unique()):
 #           print(f'{ii_cfgc/len([x for x in df_onchip.cfg.unique()]):.3f}',end='\r')
 #           area, latency_cc, energy = 0, 0, 0
 #           total_weight_mem,act_mem=0,0
 #           energy_breakdown = {'IMC':0,'W':0,'I':0,'O':0}
 #           utilization,spat_utilization, temp_utilization, total_MAC_count = 0,0,0,0
 #           for ii_layer, layer in enumerate(dfx.layer.unique()):
 #               D1 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D1']
 #               D2 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D2']
 #               D3 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D3']
 #               M = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['M']
 #           #    print(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['loop_dim_size'])
 #               ll = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['loop_dim_size']
 #               area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['area']
 #               area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_sram_area']
 #               area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['act_sram_area']
 #               weight_sram_area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_sram_area']
 #               act_sram_area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['act_sram_area']
 #               latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['latency_cc']
 #               latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_loading_cc']
 #               energy += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
 #               energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
 #               energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
 #               energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][:-1])
 #               energy_breakdown['IMC'] += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
 #               energy_breakdown['I'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
 #               energy_breakdown['O'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
 #               energy_breakdown['W'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][:-1])
 #               try:
 #                   rx = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]
 #                   spat_utilization += rx['MAC_spatial_utilization'] * rx['total_MAC_count']
 #                   temp_utilization += (rx['latency_cc'] / (rx['latency_cc'] + rx['weight_loading_cc'])) * rx['total_MAC_count']
 #                   utilization += rx['MAC_spatial_utilization'] * (rx['latency_cc'] / (rx['latency_cc'] + rx['weight_loading_cc'] )) * rx['total_MAC_count']
 #                   total_MAC_count += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['total_MAC_count']
 #               except:
 #                   breakpoint()

 #           total_points.append({'cfg':cfgc,'area':area,'latency_total':latency_cc, 'energy_total':energy, 'utilization':utilization, 'total_MAC_count': total_MAC_count,'spat_utilization':spat_utilization,
 #               'temp_utilization':temp_utilization,
 #               'D1':D1,'D2':D2,'D3':D3,'M':M, 'weight_sram_area':weight_sram_area, 'act_sram_area':act_sram_area, 'energy_breakdown':energy_breakdown})
 #       df_onchip_sram = pd.DataFrame(total_points)

 #       with open(f"outputs_{network}/{imc_type}/var/onchip_weights_{network}.pkl","wb") as infile:
 #           pickle.dump(df_onchip_sram, infile)

        # All weights on on-chip SRAM + DOUBLE BUFFERING
        df_onchip = df.copy(deep=True)
        dfx = df_onchip
        total_points = []
        for ii_cfgc, cfgc in enumerate(df_onchip.cfg.unique()):
            print(f'{ii_cfgc/len([x for x in df_onchip.cfg.unique()]):.3f}',end='\r')
            area, latency_cc, energy = 0, 0, 0
            total_weight_mem,act_mem=0,0
            energy_breakdown = {'IMC':0,'W':0,'I':0,'O':0}
            utilization,spat_utilization, temp_utilization, total_MAC_count = 0,0,0,0
            for ii_layer, layer in enumerate(dfx.layer.unique()):
                D1 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D1']
                D2 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D2']
                D3 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D3']
                M = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['M']
            #    print(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['loop_dim_size'])
                ll = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['loop_dim_size']
                area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['area_db']
                area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_sram_area']
                area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['act_sram_area']
                weight_sram_area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_sram_area']
                act_sram_area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['act_sram_area']
                latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['latency_cc']
                latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_loading_cc_db']
                energy += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][:-1])
                energy_breakdown['IMC'] += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
                energy_breakdown['I'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
                energy_breakdown['O'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
                energy_breakdown['W'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][:-1])
                try:
                    rx = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]
                    spat_utilization += rx['MAC_spatial_utilization'] * rx['total_MAC_count']
                    temp_utilization += (rx['latency_cc'] / (rx['latency_cc'])) * rx['total_MAC_count']
                    utilization += rx['MAC_spatial_utilization'] * (rx['latency_cc'] / (rx['latency_cc'])) * rx['total_MAC_count']
                    total_MAC_count += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['total_MAC_count']
                except:
                    breakpoint()

            total_points.append({'cfg':cfgc,'area':area,'latency_total':latency_cc, 'energy_total':energy, 'utilization':utilization, 'total_MAC_count': total_MAC_count,'spat_utilization':spat_utilization,
                'temp_utilization':temp_utilization,
                'D1':D1,'D2':D2,'D3':D3,'M':M, 'weight_sram_area':weight_sram_area, 'act_sram_area':act_sram_area, 'energy_breakdown':energy_breakdown})
        df_onchip_sram = pd.DataFrame(total_points)

        with open(f"outputs_{network}/{imc_type}/var/onchip_weights_{network}_db.pkl","wb") as infile:
            pickle.dump(df_onchip_sram, infile)
        continue

        # All weights off-chip
        df_offchip = df.copy(deep=True)
        dfx = df_offchip
        total_points = []
        for ii_cfgc, cfgc in enumerate(df_offchip.cfg.unique()):
            print(f'{ii_cfgc/len([x for x in df_offchip.cfg.unique()]):.3f}',end='\r')
            area, latency_cc, energy = 0, 0, 0
            energy_breakdown = {'IMC':0,'W':0,'I':0,'O':0}
            utilization,spat_utilization, temp_utilization, total_MAC_count = 0,0,0,0
            for ii_layer, layer in enumerate(dfx.layer.unique()):
                D1 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D1']
                D2 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D2']
                D3 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D3']
                M = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['M']
                area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['area']
                area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['act_sram_area']
                weight_sram_area = 0
                act_sram_area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['act_sram_area']
                latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['latency_cc']
                latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_loading_cc']
                latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['dram_loading_weight']
                latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['dram_loading_SS']
                energy += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
                energy += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][0]
                energy += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][-1]
                energy_breakdown['IMC'] += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
                energy_breakdown['I'] +=sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
                energy_breakdown['O'] +=sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
                energy_breakdown['W'] +=dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][-1]
                energy_breakdown['W'] +=dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][0]
                try:
                    rx = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]
                    spat_utilization += rx['MAC_spatial_utilization'] * rx['total_MAC_count']
                    temp_utilization += (rx['latency_cc'] / (rx['latency_cc'] + rx['weight_loading_cc'] + rx['dram_loading_weight'] + rx['dram_loading_SS'])) * rx['total_MAC_count']
                    utilization += rx['MAC_spatial_utilization'] * (rx['latency_cc'] / (rx['latency_cc'] + rx['weight_loading_cc'] + rx['dram_loading_weight'] + rx['dram_loading_SS'])) * rx['total_MAC_count']
                    total_MAC_count += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['total_MAC_count']
                except:
                    breakpoint()


            total_points.append({'cfg':cfgc,'area':area,'latency_total':latency_cc, 'energy_total':energy, 'utilization':utilization, 'total_MAC_count': total_MAC_count,'spat_utilization':spat_utilization,
                'temp_utilization':temp_utilization,
                'D1':D1,'D2':D2,'D3':D3,'M':M, 'weight_sram_area':weight_sram_area, 'act_sram_area':act_sram_area, 'energy_breakdown':energy_breakdown})
        df_offchip_sram = pd.DataFrame(total_points)

        with open(f"outputs_{network}/{imc_type}/var/offchip_weights_{network}.pkl","wb") as infile:
            pickle.dump(df_offchip_sram, infile)


if __name__ == "__main__":
    fig_plot()
