import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import pandas as pd
import pickle
import os
from copy import deepcopy
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import plotly.express as px
import shutil
from plotly.subplots import make_subplots
import matplotlib.patheffects as pe

import zigzag
colors = ["#51a3a3","#75485e","#cb904d","#af9ab2","#cbe896"]
def get_df(directory, network):
    power_values, area_values = [], []
    df = pd.DataFrame()
    ii  = 0
    for filename in os.listdir(directory):
        ii+=1
        if filename.split(".")[-1] != 'pkl':
            continue
        if filename != f'{network}_all_data_points.pkl':
            continue
        f = os.path.join(directory, filename)
        
        with open(f, 'rb') as infile:
            data = pickle.load(infile)
        df_data = []
        for cc in data:
            for l in cc:
                if l.layer.id != 4:
                    continue
                all_weight_unrolled = False
                weight_tm = [j for i in l.temporal_mapping.mapping_dic_origin['W'] for j in i]
                if all([x[0] not in ['K','C','FX','FY'] for x in weight_tm]):
                    all_weight_unrolled = True
                df_data.append({'layer':l.layer.id,
                    'loop_dim_size':l.layer.loop_dim_size,
                    'area':l.imc_area,
                    'imc_area_breakdown':l.imc_area_breakdown,
                    'weight_sram_area':l.mem_area_breakdown['weight_sram'],
                    'act_sram_area':l.mem_area_breakdown['sram_256KB'],
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


def get_df_pkl(directory, network):
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
        data = [data]
        for cc in data:
            for l in cc:
#                if l.layer.id != 4:
#                    continue
                all_weight_unrolled = False
                weight_tm = [j for i in l.temporal_mapping.mapping_dic_origin['W'] for j in i]
                if all([x[0] not in ['K','C','FX','FY'] for x in weight_tm]):
                    all_weight_unrolled = True
                df_data.append({'layer':l.layer.id,
                    'loop_dim_size':l.layer.loop_dim_size,
                    'area':l.imc_area,
                    'imc_area_breakdown':l.imc_area_breakdown,
                    'weight_sram_area':l.mem_area_breakdown['weight_sram'],
                    'act_sram_area':l.mem_area_breakdown['sram_256KB'],
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

def fig_D1D2_comparison():
    network_list = ['deepautoencoder']
    imc_type = 'dimc' 
    for network in network_list:
        df = get_df(f"./outputs_{network}/{imc_type}/", network)
        # All weights on on-chip SRAM
        df_onchip = df.copy(deep=True)
        dfx = df_onchip
        total_points = []
        for ii_cfgc, cfgc in enumerate(df_onchip.cfg.unique()):
            print(f'{ii_cfgc/len([x for x in df_onchip.cfg.unique()]):.3f}',end='\r')
            area, latency_cc, energy = 0, 0, 0
            total_weight_mem,act_mem=0,0
            utilization, total_MAC_count = 0,0
            energy_breakdown = {'IMC':0,'W':0,'I':0,'O':0}
            tclk_breakdown = {}
            for ii_layer, layer in enumerate(dfx.layer.unique()):
                if layer != 4:
                    continue
                D1 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D1']
                D2 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D2']
                D3 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D3']
                M = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['M']
                num_cells = D1 * D2 * D3 * M
                ll = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['loop_dim_size']
                area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['area']
                area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_sram_area']
                area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['act_sram_area']
                weight_sram_area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_sram_area']
                act_sram_area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['act_sram_area']
                latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['latency_cc']
                latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_loading_cc']
                energy += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][:-1])
                energy_breakdown['IMC'] += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
                energy_breakdown['I'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
                energy_breakdown['O'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
                energy_breakdown['W'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][:-1])
                tclk_breakdown = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['tclk_breakdown']
                imc_area_breakdown = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['imc_area_breakdown']
                imc_energy_breakdown = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown']
                try:
                    utilization += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_spatial_utilization'] * dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['total_MAC_count']
                    total_MAC_count += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['total_MAC_count']
                except:
                    breakpoint()
            total_points.append({'cfg':cfgc,'area':area,'latency_total':latency_cc, 'energy_total':energy, 'num_cells':num_cells, 'utilization':utilization, 'total_MAC_count': total_MAC_count,'imc_energy_breakdown':imc_energy_breakdown,
                'D1':D1,'D2':D2,'D3':D3,'M':M, 'weight_sram_area':weight_sram_area, 'act_sram_area':act_sram_area, 'energy_breakdown':energy_breakdown, 'tclk_breakdown':tclk_breakdown, 'imc_area_breakdown':imc_area_breakdown})
        df_onchip_sram = pd.DataFrame(total_points)
        dfxx = df_onchip_sram
        dxxx = 32
        dfx = dfxx[(dfxx.M == 1)]
        dfx['EDP'] = dfx['latency_total'] * dfx['energy_total']
        breakpoint()
        tclk_breakdown = []
        for i,r in dfx.iterrows():
            for k,v in r.tclk_breakdown.items():
                data = {}
                data['type'] = k
                data['value'] = v
                data['M'] = r.M
                tclk_breakdown.append(data)
        tclk_breakdown = pd.DataFrame(tclk_breakdown)

        imc_area_breakdown = []
        for i,r in dfx.iterrows():
            for k,v in r.imc_area_breakdown.items():
                data = {}
                data['type'] = k
                data['value'] = v
                data['M'] = r.M
                imc_area_breakdown.append(data)
        imc_area_breakdown = pd.DataFrame(imc_area_breakdown)

        imc_energy_breakdown = []
        for i,r in dfx.iterrows():
            for k,v in r.imc_energy_breakdown.items():
                data = {}
                data['type'] = k
                data['value'] = v
                data['M'] = r.M
                imc_energy_breakdown.append(data)
        imc_energy_breakdown = pd.DataFrame(imc_energy_breakdown)

        latency_breakdown = []
        for i,r in dfx.iterrows():
            data = {}
            data['type'] = 'latency' 
            data['value'] = r.latency_total
            data['M'] = r.M
            latency_breakdown.append(data)
        latency_breakdown = pd.DataFrame(latency_breakdown)

        edp = []
        for i,r in dfx.iterrows():
            data = {}
            data['type'] = 'EDP' 
            data['value'] = r.EDP
            data['M'] = r.M
            edp.append(data)
        edp = pd.DataFrame(edp)



        fig, ax = plt.subplots()
        ax.set_axisbelow(True)
        bottom = np.zeros(len([x for x in tclk_breakdown.M.unique()]))
        tclk_breakdown =tclk_breakdown.sort_values(by=['M'], ascending=True,ignore_index=True)
        m_vals = [0] + [str(x) for x in tclk_breakdown.M.unique()]
        for k in ['accumulators','adders','mults','group_depth']:
            df = tclk_breakdown[tclk_breakdown.type == k]
            ax.bar(range(len(df['M'].unique())), df['value'], label=k, bottom=bottom)
            bottom += df['value'].to_numpy()
        ax.set_xticklabels(m_vals)
        plt.legend()
        plt.xlabel('M factor')
        plt.ylabel('Max $t_{clk}$ [ns]')
        plt.grid()
        plt.tight_layout()
        plt.show()
        breakpoint()

        fig, ax = plt.subplots()
        bottom = np.zeros(len([x for x in imc_area_breakdown.M.unique()]))
        ax.set_axisbelow(True)
        imc_area_breakdown =imc_area_breakdown.sort_values(by=['M'], ascending=True,ignore_index=True)
        for k in ['accumulators','adders','mults','cells']:
            df = imc_area_breakdown[imc_area_breakdown.type == k]
            ax.bar(range(len(df['M'].unique())), df['value'], label=k, bottom=bottom)
            bottom += df['value'].to_numpy()
        ax.set_xticklabels(m_vals)
        plt.legend()
        plt.xlabel('M factor')
        plt.ylabel('Area breakdown [$\mu$m$^2$]')
        plt.grid()
        plt.tight_layout()
        plt.show()
        breakpoint()

        fig, ax = plt.subplots()
        bottom = np.zeros(len([x for x in imc_energy_breakdown.M.unique()]))
        ax.set_axisbelow(True)
        imc_energy_breakdown =imc_energy_breakdown.sort_values(by=['M'], ascending=True,ignore_index=True)
        for k in ['accumulators','adders','mults','precharging']:
            df = imc_energy_breakdown[imc_energy_breakdown.type == k]
            ax.bar(range(len(df['M'].unique())), df['value'], label=k, bottom=bottom)
            bottom += df['value'].to_numpy()
        ax.set_xticklabels(m_vals)
        plt.legend()
        plt.xlabel('M factor')
        plt.ylabel('Energy breakdown [pJ]')
        plt.grid()
        plt.tight_layout()
        plt.show()
        breakpoint()
     
        fig, ax = plt.subplots()
        bottom = np.zeros(len([x for x in latency_breakdown.M.unique()]))
        ax.set_axisbelow(True)
        imc_energy_breakdown =imc_energy_breakdown.sort_values(by=['M'], ascending=True,ignore_index=True)
        for k in ['latency']:
            df = latency_breakdown[latency_breakdown.type == k]
            ax.bar(range(len(df['M'].unique())), df['value'], label=k, bottom=bottom)
            bottom += df['value'].to_numpy()
        ax.set_xticklabels(m_vals)
        plt.legend()
        plt.xlabel('M factor')
        plt.ylabel('Latency [clock cycles]')
        plt.grid()
        plt.tight_layout()
        plt.show()
        breakpoint()
 
        fig, ax = plt.subplots()
        bottom = np.zeros(len([x for x in edp.M.unique()]))
        ax.set_axisbelow(True)
        imc_energy_breakdown =imc_energy_breakdown.sort_values(by=['M'], ascending=True,ignore_index=True)
        for k in ['EDP']:
            df = edp[edp.type == k]
            ax.bar(range(len(df['M'].unique())), df['value'], label=k, bottom=bottom)
            bottom += df['value'].to_numpy()
        ax.set_xticklabels(m_vals)
        plt.legend()
        plt.xlabel('M factor')
        plt.ylabel('EDP [pJ $\times$ clock cycles]')
        plt.grid()
        plt.tight_layout()
        plt.show()
        breakpoint()
 
def fig_M_comparison():
    network_list = ['resnet8']
    imc_type = 'dimc' 
    for network in network_list:
        df = get_df(f"./outputs_{network}/{imc_type}/", network)
        # All weights on on-chip SRAM
        df_onchip = df.copy(deep=True)
        dfx = df_onchip
        total_points = []
        for ii_cfgc, cfgc in enumerate(df_onchip.cfg.unique()):
            print(f'{ii_cfgc/len([x for x in df_onchip.cfg.unique()]):.3f}',end='\r')
            area, latency_cc, energy = 0, 0, 0
            total_weight_mem,act_mem=0,0
            utilization, total_MAC_count = 0,0
            energy_breakdown = {'IMC':0,'W':0,'I':0,'O':0}
            tclk_breakdown = {}
            for ii_layer, layer in enumerate(dfx.layer.unique()):
                if layer != 4:
                    continue
                D1 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D1']
                D2 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D2']
                D3 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D3']
                M = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['M']
                num_cells = D1 * D2 * D3 * M
                ll = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['loop_dim_size']
                area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['area']
                area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_sram_area']
                area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['act_sram_area']
                weight_sram_area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_sram_area']
                act_sram_area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['act_sram_area']
                latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['latency_cc']
                latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_loading_cc']
                energy += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][:-1])
                energy_breakdown['IMC'] += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
                energy_breakdown['I'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
                energy_breakdown['O'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
                energy_breakdown['W'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][:-1])
                tclk_breakdown = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['tclk_breakdown']
                imc_area_breakdown = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['imc_area_breakdown']
                imc_energy_breakdown = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown']
                try:
                    utilization += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_spatial_utilization'] * dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['total_MAC_count']
                    total_MAC_count += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['total_MAC_count']
                except:
                    breakpoint()
            total_points.append({'cfg':cfgc,'area':area,'latency_total':latency_cc, 'energy_total':energy, 'num_cells':num_cells, 'utilization':utilization, 'total_MAC_count': total_MAC_count,'imc_energy_breakdown':imc_energy_breakdown,
                'D1':D1,'D2':D2,'D3':D3,'M':M, 'weight_sram_area':weight_sram_area, 'act_sram_area':act_sram_area, 'energy_breakdown':energy_breakdown, 'tclk_breakdown':tclk_breakdown, 'imc_area_breakdown':imc_area_breakdown})
        df_onchip_sram = pd.DataFrame(total_points)
        dfxx = df_onchip_sram
        dfxx['EDP'] = dfxx['latency_total'] * dfxx['energy_total']

#        for h in dfgg.num_cells.unique():
#            dfgx = dfgg[dfgg.num_cells == h]
#            for d1 in dfgx.D1.unique():
#                dfgxx = dfgx[dfgx.D1 == d1]
#                dfgxx = dfgxx.sort_values(by=['D3'],ignore_index=True)
#                d3_unique = dfgxx.D3.unique()
#                if len(d3_unique) >= 3 and 1 in d3_unique:
#                    fig = px.bar(dfgxx, 'D3_str','latency_total',log_y=True,title=h)
#                    fig.show()
#
#        breakpoint()
        dxxx = 32
        dfx = dfxx[(dfxx.num_cells == dxxx**2) & (dfxx.D2 == dxxx) & (dfxx.D3 == 1)]
        tclk_breakdown = []
        for i,r in dfx.iterrows():
            for k,v in r.tclk_breakdown.items():
                data = {}
                data['type'] = k
                data['value'] = v
                data['M'] = r.M
                tclk_breakdown.append(data)
        tclk_breakdown = pd.DataFrame(tclk_breakdown)

        imc_area_breakdown = []
        for i,r in dfx.iterrows():
            for k,v in r.imc_area_breakdown.items():
                data = {}
                data['type'] = k
                data['value'] = v
                data['M'] = r.M
                imc_area_breakdown.append(data)
        imc_area_breakdown = pd.DataFrame(imc_area_breakdown)

        imc_energy_breakdown = []
        for i,r in dfx.iterrows():
            for k,v in r.imc_energy_breakdown.items():
                data = {}
                data['type'] = k
                data['value'] = v
                data['M'] = r.M
                imc_energy_breakdown.append(data)
        imc_energy_breakdown = pd.DataFrame(imc_energy_breakdown)

        latency_breakdown = []
        for i,r in dfx.iterrows():
            data = {}
            data['type'] = 'latency' 
            data['value'] = r.latency_total
            data['M'] = r.M
            latency_breakdown.append(data)
        latency_breakdown = pd.DataFrame(latency_breakdown)

        edp = []
        for i,r in dfx.iterrows():
            data = {}
            data['type'] = 'EDP' 
            data['value'] = r.EDP
            data['M'] = r.M
            edp.append(data)
        edp = pd.DataFrame(edp)

        tclk_breakdown = tclk_breakdown.sort_values(by=['M'],ignore_index=True)
        m_vals = [str(x) for x in tclk_breakdown.M.unique()]

        fig, ax = plt.subplots(2,2,figsize=(6,4))
        ax[0][0].set_axisbelow(True)
        bottom = np.zeros(len([x for x in tclk_breakdown.M.unique()]))
        tclk_breakdown =tclk_breakdown.sort_values(by=['M'], ascending=True,ignore_index=True)
        for ii, k in enumerate(['accumulators','adders','mults','group_depth']):
            df = tclk_breakdown[tclk_breakdown.type == k]
            ax[0][0].bar(range(len(df['M'].unique())), df['value'], label=k, bottom=bottom, color=colors[ii])
            bottom += df['value'].to_numpy()
        ax[0][0].set_xticks([0,1,2,3,4,5])
        ax[0][0].set_xticklabels(m_vals)
        ax[0][0].ticklabel_format(axis='y',style='sci')
        ax[0][0].set_xlabel('M factor')
        ax[0][0].set_ylabel('Max $t_{clk}$ [ns]')
        ax[0][0].grid()

        bottom = np.zeros(len([x for x in imc_area_breakdown.M.unique()]))
        ax[1][0].set_axisbelow(True)
        imc_area_breakdown =imc_area_breakdown.sort_values(by=['M'], ascending=True,ignore_index=True)
        for ii, k in enumerate(['accumulators','adders','mults','cells']):
            df = imc_area_breakdown[imc_area_breakdown.type == k]
            ax[1][0].bar(range(len(df['M'].unique())), df['value'], label=k, bottom=bottom, color=colors[ii])
            bottom += df['value'].to_numpy()
        ax[1][0].set_xticks([0,1,2,3,4,5])
        ax[1][0].set_xticklabels(m_vals)
        ax[1][0].ticklabel_format(axis='y',style='sci')
        ax[1][0].legend()
        ax[1][0].set_xlabel('M factor')
        ax[1][0].set_ylabel('Area breakdown [$\mu$m$^2$]')
        ax[1][0].grid()

        bottom = np.zeros(len([x for x in imc_energy_breakdown.M.unique()]))
        ax[0][1].set_axisbelow(True)
        imc_energy_breakdown =imc_energy_breakdown.sort_values(by=['M'], ascending=True,ignore_index=True)
        imc_energy_breakdown['value'] /= 1e6
        imc_energy_breakdown.loc[imc_energy_breakdown.type == 'adders','value'] = 0.557830
        for ii,k in enumerate(['accumulators','adders','mults','precharging']):
            df = imc_energy_breakdown[imc_energy_breakdown.type == k]
            ax[0][1].bar(range(len(df['M'].unique())), df['value'], label=k, bottom=bottom, color=colors[ii])
            bottom += df['value'].to_numpy()
        ax[0][1].set_xticks([0,1,2,3,4,5])
        ax[0][1].set_xticklabels(m_vals)
        ax[0][1].ticklabel_format(axis='y',style='scientific')
        ax[0][1].set_xlabel('M factor')
        ax[0][1].set_ylabel('Energy breakdown [$\mu$J]')
        ax[0][1].grid()
     
        #bottom = np.zeros(len([x for x in latency_breakdown.M.unique()]))
        #ax[0][0].set_axisbelow(True)
        #latency_breakdown = latency_breakdown.sort_values(by=['M'], ascending=True,ignore_index=True)
        #for k in ['latency']:
        #    df = latency_breakdown[latency_breakdown.type == k]
        #    ax[0][0].bar(range(len(df['M'].unique())), df['value'], label=k, bottom=bottom)
        #    bottom += df['value'].to_numpy()
        #ax[0][0].set_xticklabels(m_vals)
        #ax[0][0].legend()
        #ax[0][0].xlabel('M factor')
        #plt.ylabel('Latency [clock cycles]')
        #ax[0][0].grid()
        #ax[0][0].tight_layout()
        #ax[0][0].show()
        #breakpoint()
 
        bottom = np.zeros(len([x for x in edp.M.unique()]))
        ax[1][1].set_axisbelow(True)
        edp = edp.sort_values(by=['M'], ascending=True,ignore_index=True)
        for k in ['EDP']:
            df = edp[edp.type == k]
            ax[1][1].bar(range(len(df['M'].unique())), df['value'], label=k, bottom=bottom, color=colors[ii])
            bottom += df['value'].to_numpy()
        ax[1][1].set_xticks([0,1,2,3,4,5])
        ax[1][1].set_xticklabels(m_vals)
        ax[1][1].set_xlabel('M factor')
        ax[1][1].ticklabel_format(axis='y',style='sci')
        ax[1][1].set_ylabel(r'EDP [pJ $ \times $ clock cycles]')
        ax[1][1].grid()
        plt.tight_layout()
        plt.show()
        breakpoint()
 
def fig_d3_comparison():
    network_list = ['resnet8']
    imc_type = 'dimc' 
    for network in network_list:
        df = get_df_pkl(f"./outputs_{network}/{imc_type}/d3_experiment", network)
       # All weights on on-chip SRAM
        df_onchip = df.copy(deep=True)
        dfx = df_onchip
        total_points = []
        for ii_cfgc, cfgc in enumerate(df_onchip.cfg.unique()):
            print(f'{ii_cfgc/len([x for x in df_onchip.cfg.unique()]):.3f}',end='\r')
            area, latency_cc, energy = 0, 0, 0
            total_weight_mem,act_mem=0,0
            utilization, total_MAC_count = 0,0
            energy_breakdown = {'IMC':0,'W':0,'I':0,'O':0}
            tclk_breakdown = {}
            for ii_layer, layer in enumerate(dfx.layer.unique()):
                if layer != 4:
                    continue
                D1 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D1']
                D2 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D2']
                D3 = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['D3']
                M = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['M']
                num_cells = D1 * D2 * D3 * M
                ll = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['loop_dim_size']
                area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['area']
                area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_sram_area']
                area += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['act_sram_area']
                weight_sram_area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_sram_area']
                act_sram_area = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['act_sram_area']
                latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['latency_cc']
                latency_cc += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['weight_loading_cc']
                energy += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
                energy += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][:-1])
                energy_breakdown['IMC'] += sum([x for x in dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown'].values()])
                energy_breakdown['I'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['I'][:-1])
                energy_breakdown['O'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['O'][:-1])
                energy_breakdown['W'] += sum(dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['mem_energy_breakdown']['W'][:-1])
                tclk_breakdown = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['tclk_breakdown']
                imc_area_breakdown = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['imc_area_breakdown']
                imc_energy_breakdown = dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_energy_breakdown']
                try:
                    utilization += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['MAC_spatial_utilization'] * dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['total_MAC_count']
                    total_MAC_count += dfx[(dfx.layer == layer) & (dfx.cfg == cfgc)].iloc[0]['total_MAC_count']
                except:
                    breakpoint()
            total_points.append({'cfg':cfgc,'area':area,'latency_total':latency_cc, 'energy_total':energy, 'num_cells':num_cells, 'utilization':utilization, 'total_MAC_count': total_MAC_count,'imc_energy_breakdown':imc_energy_breakdown,
                'D1':D1,'D2':D2,'D3':D3,'M':M, 'weight_sram_area':weight_sram_area, 'act_sram_area':act_sram_area, 'energy_breakdown':energy_breakdown, 'tclk_breakdown':tclk_breakdown, 'imc_area_breakdown':imc_area_breakdown})
        df_onchip_sram = pd.DataFrame(total_points)
        dfxx = df_onchip_sram
        dfxx['EDP'] = dfxx['latency_total'] * dfxx['energy_total']
        breakpoint()


def fig_pareto_plot_m(parameter=None):
    area_net = {'mobilenet_v1':{'act':0.157,'weight':0.59},
            'resnet8':{'act':0.093, 'weight':0.213},
            'deepautoencoder':{'act':0.0058, 'weight':0.8},
            'ds_cnn':{'act':0.046, 'weight':0.066},
            'geom_mean':{'act':0.147, 'weight':0.213},
            'weighted_geom_mean':{'act':0.147, 'weight':0.213}}
            
    imc_type = 'dimc'
    #for parameter in ['latency_total','energy_total','EDP']:
    for parameter in ['EDP']:
        df_total = pd.DataFrame()
        df_pareto_total = pd.DataFrame()
        fig,ax = plt.subplots(2,2, figsize=(7,5))
        for no_d3 in [True]:
            for network in ['resnet8','deepautoencoder','mobilenet_v1','ds_cnn']:
                with open(f"outputs_{network}/{imc_type}/var/onchip_weights_{network}.pkl","rb") as infile:
                    df_onchip_weights = pickle.load(infile)
                df_onchip_weights['EDP'] = df_onchip_weights['latency_total'] * df_onchip_weights['energy_total']
                if no_d3:
                    df_onchip_weights = df_onchip_weights[df_onchip_weights.D3 == 1]
                    df_onchip_weights['type'] = 'weights_on_chip no_d3'
                else:
                    df_onchip_weights = df_onchip_weights[df_onchip_weights.D3 != 1]
                    df_onchip_weights['type'] = 'weights_on_chip d3'
                df_onchip_weights = df_onchip_weights.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)
                df_onchip_pareto = pd.DataFrame()
                for m in df_onchip_weights.M.unique():
                    dfxx = df_onchip_weights[df_onchip_weights.M == m]
                    best_latency = float('inf')
                    for i,r in dfxx.iterrows():
                        if r[parameter] < best_latency:
                            df_onchip_pareto = df_onchip_pareto.append(r)
                            best_latency = r[parameter]
                if no_d3:
                    df_onchip_pareto['type'] = 'weights_on_chip no_d3'
                else:
                    df_onchip_pareto['type'] = 'weights_on_chip d3'
                df_onchip_pareto = df_onchip_pareto.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)
                
                df = pd.concat([df_onchip_weights])
                df_pareto = pd.concat([df_onchip_pareto])
                df['network'] = network
                df_pareto['network'] = network
                df_pareto['parameter'] = parameter        
                df_total['parameter'] = parameter
                if df_total.empty:
                    df_total = df.copy()
                else:
                    df_total = pd.concat([df, df_total])
                
                if df_pareto_total.empty:
                    df_pareto_total = df_pareto.copy()
                else:
                    df_pareto_total = pd.concat([df_pareto, df_pareto_total])
            with open(f'outputs_{network}/{imc_type}/var/tinymlperf_pareto_{parameter}.pkl','wb') as infile:
                pickle.dump(df_pareto_total,infile)
        

        for type in df_total.type.unique():
            for cfg in df_total.cfg.unique():
                dfx = df_total[(df_total.cfg == cfg) & (df_total.type == type)]
                dfx_networks = [x for x in dfx.network.unique()]
                if all([x in dfx_networks for x in ['mobilenet_v1','deepautoencoder','resnet8','ds_cnn']]):
                    geom_mean = 1
                    weighted_geom_mean = 0
                    total_MAC_count = 0
                    for i,r in dfx.iterrows():
                        if r.network not in ['geom_mean','weighted_geom_mean']:
                            geom_mean *= r[parameter]
                            weighted_geom_mean += np.log(r[parameter]) * r.total_MAC_count
                            total_MAC_count += r.total_MAC_count
                    geom_mean = np.power(geom_mean, 1/4)
                    weighted_geom_mean = np.exp(weighted_geom_mean/total_MAC_count)
                    if 'geom_mean' not in dfx_networks:
                        r_gm = deepcopy(dfx.iloc[0])
                        r_wgm = deepcopy(dfx.iloc[0])
                        r_gm.network = 'geom_mean'
                        r_gm[parameter] = geom_mean
                        if type.split(' ')[0] == 'dram_weights':
                            r_gm['area'] = r_gm.area - r_gm.weight_sram_area - r_gm.act_sram_area + 0.157
                            r_wgm['area'] = r_wgm.area - r_wgm.weight_sram_area - r_wgm.act_sram_area + 0.157
                        else:
                            r_gm['area'] = r_gm.area - r_gm.weight_sram_area - r_gm.act_sram_area + 0.157+0.8
                            r_wgm['area'] = r_wgm.area - r_wgm.weight_sram_area - r_wgm.act_sram_area + 0.157+0.8

                        r_wgm.network = 'weighted_geom_mean'
                        r_wgm[parameter] = weighted_geom_mean
                        df_total = df_total.append(r_gm)#, ignore_index=True)
                        df_total = df_total.append(r_wgm)#, ignore_index=True)
                    else:
                        df_total.loc[(df_total.type == type) & (df_total.cfg == cfg) & (df_total.network == 'geom_mean'), parameter] = geom_mean
                        df_total.loc[(df_total.type == type) & (df_total.cfg == cfg) & (df_total.network == 'weighted_geom_mean'), parameter] = weighted_geom_mean


            df_total = df_total.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)
            for  netw_gm in ['geom_mean','weighted_geom_mean']:
                dfx = df_total[(df_total.type == type) & (df_total.network == netw_gm)]
                df_gm_pareto = pd.DataFrame()
                best_gm = float('inf')
                for i,r in dfx.iterrows():
                    if r[parameter] < best_gm:
                        df_gm_pareto = df_gm_pareto.append(r)
                        best_gm = r[parameter]
                df_gm_pareto['parameter'] = parameter
                df_gm_pareto['type'] = type
                df_gm_pareto['network'] = netw_gm
                df_pareto_total = pd.concat([df_pareto_total, df_gm_pareto])

        df_pareto_total['M'] = df_pareto_total['M'].astype(str)
        df_total['d1d2'] = 0
        for i,r in df_total.iterrows():
            df_total.loc[i,'d1d2'] = f'{r["D1"]} {r["D2"]}'
        breakpoint()

        fig = px.scatter(df_pareto_total, 'area', parameter, color='network', symbol='type',log_x=True,log_y=True) 
        
        fig.show()
        #breakpoint()
        

        lines = {0:'-',1:':',2:'--',3:'-.'}
        colors= plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig = make_subplots(2,2,subplot_titles=[x for x in df_total.network.unique()])
        df_pareto_total_tmp = df_pareto_total.copy(deep=True)
        df_pareto_total = df_pareto_total.sort_values(by=['area'],ascending=[True],ignore_index=True)
        df_pareto_total = df_pareto_total[(df_pareto_total.network != 'geom_mean') & (df_pareto_total.network != 'weighted_geom_mean')]
        for ii_network, network in enumerate(df_pareto_total.network.unique()):
            ax[ii_network].set_title(network)
            ax[ii_network].grid()
            if ii_network %2 > 0:
                ax[ii_network].set_xlabel('Area [$\mu$m$^2$]')
            if ii_network //2 == 0:
                ax[ii_network].set_ylabel(parameter)
            ax[ii_network].axvline(x=area_net[network]['act'],color='tomato',lw=4)
            ax[ii_network].axvline(x=area_net[network]['weight']+area_net[network]['act'],color='darkkhaki',lw=4)
            for type in df_pareto_total.type.unique():
                if type.split(' ')[0] == 'dram_weights':
                    ii_c = 0
                elif type.split(' ')[0] == 'weights_on_chip':
                    ii_c = 1
                else:
                    ii_c = 2
                if type.split(' ')[1] == 'no_d3':
                    ls = 0
                else:
                    ls = 1
                
                dfx =  df_total[(df_total.network == network) & (df_total.type == type)]
                dfx_pareto =  df_pareto_total[(df_pareto_total.network == network) & (df_pareto_total.type == type)]
                ax[ii_network].scatter(dfx['area'],dfx[parameter],label=type,c=colors[ii_c],alpha=0.2)
                ax[ii_network].plot(dfx_pareto['area'],dfx_pareto[parameter],c=colors[ii_c],ls=lines[ls],lw=3,marker='o',markersize=3, label=type.split(' ')[0])
#                for i,r in dfx_pareto.iterrows():
#                    if i%2 == 0:
#                        ax[ii_network].text(r['area'],r[parameter],str(int(r['M'])), color='black',path_effects=[pe.withStroke(linewidth=4, foreground='w')])#bbox={'facecolor':colors[ii_c],'pad':1,'edgecolor':None})
                ax[ii_network].set_yscale('log')
                ax[ii_network].set_xscale('log')
            #    fig.add_trace(go.Scatter(x = dfx['area'],y=dfx[parameter],mode='markers',name=f'{network} {type}'), col=(ii_network%2)+1, row=(ii_network//2)+1)
            #    fig.add_trace(go.Scatter(x = dfx_pareto['area'],y=dfx_pareto[parameter],mode='markers+lines',name=f'{network} {type} pareto'), col=(ii_network%2)+1, row=(ii_network//2)+1)
#            ax[ii_network].legend()
           # ax[ii_network].set_xlim(right=20)
        plt.tight_layout()
        plt.show()
        #breakpoint()
        #fig.update_xaxes(type='log')
        #fig.update_yaxes(type='log')
        #fig.show()
        dfx_pareto = df_pareto_total_tmp[(df_pareto_total_tmp.network == 'geom_mean') & (df_pareto_total_tmp.parameter == parameter)]
        dfx = df_total[(df_total.network == 'geom_mean') & (df_total.parameter == parameter)]
        plt.axvline(x=area_net['mobilenet_v1']['act'],color='tomato',lw=4)
        plt.axvline(x=area_net['deepautoencoder']['weight']+area_net[network]['act'],color='darkkhaki',lw=4)


        for c in dfx_pareto.type.unique():
            if c.split(' ')[0] == 'dram_weights':
                ii_c = 0
            elif c.split(' ')[0] == 'weights_on_chip':
                ii_c = 1
            else:
                ii_c = 2
            if c.split(' ')[1] == 'no_d3':
                ls = 0
            else:
                ls = 1
 
        #    dfxx =dfx[dfx.type == c]
            dfxx_pareto =dfx_pareto[dfx_pareto.type == c]
            plt.plot(dfxx_pareto['area'],dfxx_pareto[parameter],c=colors[ii_c],ls=lines[ls],lw=4,marker='o',markersize=3,label=c)
        plt.xlabel('Area [$\mu$m$^2$]')
        plt.ylabel(parameter)
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.grid()
        plt.title('Geom mean')
        plt.show()


def fig_pareto_plot_d1d2m(parameter=None):
    area_net = {'mobilenet_v1':{'act':0.157,'weight':0.59},
            'resnet8':{'act':0.093, 'weight':0.213},
            'deepautoencoder':{'act':0.0058, 'weight':0.8},
            'ds_cnn':{'act':0.046, 'weight':0.066},
            'geom_mean':{'act':0.147, 'weight':0.213},
            'weighted_geom_mean':{'act':0.147, 'weight':0.213}}
            
    imc_type = 'dimc'
    colors = ['#ff7e0e', '#5ba4bf', '#afe9af']
    #for parameter in ['latency_total','energy_total','EDP']:
    for parameter in ['EDP']:
        df_total = pd.DataFrame()
        df_pareto_total = pd.DataFrame()
        fig,ax = plt.subplots(1,4, figsize=(7,5))
        for no_d3 in [True]:
            for network in ['ds_cnn','resnet8','deepautoencoder','mobilenet_v1']:
#            for network in ['mobilenet_v1']:
                if False:#network in ['ds_cnn','resnet8','mobilenet_v1']:
                    with open(f"outputs_{network}/{imc_type}/var/imc_weights_{network}.pkl","rb") as infile:
                        df_all_weights = pickle.load(infile)

                    df_all_weights['EDP'] = df_all_weights['latency_total'] * df_all_weights['energy_total']
                    df_all_weights['type'] = 'all_weights_on_chip'
                    df_all_weights = df_all_weights.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)
                    df_aw_pareto = pd.DataFrame()
                    best_latency = float('inf')
                    for i,r in df_all_weights.iterrows():
                        if r[parameter] < best_latency:
                            df_aw_pareto = df_aw_pareto.append(r)
                            best_latency = r[parameter]
                    df_aw_pareto['type'] = 'all_weight_pareto'
                else:
                    df_aw_pareto = pd.DataFrame()
                    df_all_weights = pd.DataFrame()

                with open(f"outputs_{network}/{imc_type}/var/onchip_weights_{network}.pkl","rb") as infile:
                    df_onchip_weights = pickle.load(infile)
                df_onchip_weights['EDP'] = df_onchip_weights['latency_total'] * df_onchip_weights['energy_total']
                if no_d3:
                    df_onchip_weights = df_onchip_weights[df_onchip_weights.D3 == 1]
                    df_onchip_weights['type'] = 'weights_on_chip no_d3'
                else:
                    df_onchip_weights = df_onchip_weights[df_onchip_weights.D3 != 1]
                    df_onchip_weights['type'] = 'weights_on_chip d3'
                df_onchip_weights = df_onchip_weights.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)
                df_onchip_pareto = pd.DataFrame()
                best_latency = float('inf')
                for i,r in df_onchip_weights.iterrows():
                    if r[parameter] < best_latency:
                        df_onchip_pareto = df_onchip_pareto.append(r)
                        best_latency = r[parameter]
                if no_d3:
                    df_onchip_pareto['type'] = 'weights_on_chip no_d3'
                else:
                    df_onchip_pareto['type'] = 'weights_on_chip d3'
                df_onchip_pareto = df_onchip_pareto.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)
                with open(f"outputs_{network}/{imc_type}/var/onchip_weights_{network}_db.pkl","rb") as infile:
                    df_onchip_weights_db = pickle.load(infile)
                df_onchip_weights_db['EDP'] = df_onchip_weights_db['latency_total'] * df_onchip_weights_db['energy_total']
                if no_d3:
                    df_onchip_weights_db = df_onchip_weights_db[df_onchip_weights_db.D3 == 1]
                    df_onchip_weights_db['type'] = 'DB_weights_on_chip no_d3'
                else:
                    df_onchip_weights_db = df_onchip_weights_db[df_onchip_weights_db.D3 != 1]
                    df_onchip_weights_db['type'] = 'DB_weights_on_chip d3'
                df_onchip_weights_db = df_onchip_weights_db.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)
                df_onchip_pareto_db = pd.DataFrame()
                best_latency = float('inf')
                for i,r in df_onchip_weights_db.iterrows():
                    if r[parameter] < best_latency:
                        df_onchip_pareto_db = df_onchip_pareto_db.append(r)
                        best_latency = r[parameter]
                if no_d3:
                    df_onchip_pareto_db['type'] = 'DB_weights_on_chip no_d3'
                else:
                    df_onchip_pareto_db['type'] = 'DB_weights_on_chip d3'
                df_onchip_pareto_db = df_onchip_pareto_db.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)


                with open(f"outputs_{network}/{imc_type}/var/offchip_weights_{network}.pkl","rb") as infile:
                    df_offchip_weights = pickle.load(infile)
                df_offchip_weights['EDP'] = df_offchip_weights['latency_total'] * df_offchip_weights['energy_total']
                if no_d3:
                    df_offchip_weights = df_offchip_weights[df_offchip_weights.D3 == 1]
                    df_offchip_weights['type'] = 'dram_weights no_d3'
                else:
                    df_offchip_weights = df_offchip_weights[df_offchip_weights.D3 != 1]
                    df_offchip_weights['type'] = 'dram_weights d3'

                df_offchip_weights = df_offchip_weights.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)
                df_offchip_pareto = pd.DataFrame()
                best_latency = float('inf')
                for i,r in df_offchip_weights.iterrows():
                    if r[parameter] < best_latency:
                        df_offchip_pareto = df_offchip_pareto.append(r)
                        best_latency = r[parameter]
                if no_d3:
                    df_offchip_pareto['type'] = 'dram_weights no_d3'
                else:
                    df_offchip_pareto['type'] = 'dram_weights d3'
                df_offchip_pareto = df_offchip_pareto.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)

                df = pd.concat([df_onchip_weights, df_offchip_weights,df_onchip_weights_db, df_all_weights])
                df_pareto = pd.concat([df_onchip_pareto, df_offchip_pareto, df_onchip_pareto_db, df_aw_pareto])
                df['network'] = network
                df_pareto['network'] = network
                df_pareto['parameter'] = parameter        
                df_total['parameter'] = parameter
                if df_total.empty:
                    df_total = df.copy()
                else:
                    df_total = pd.concat([df, df_total])
                
                if df_pareto_total.empty:
                    df_pareto_total = df_pareto.copy()
                else:
                    df_pareto_total = pd.concat([df_pareto, df_pareto_total])
            with open(f'outputs_{network}/{imc_type}/var/tinymlperf_pareto_{parameter}.pkl','wb') as infile:
                pickle.dump(df_pareto_total,infile)
        

        for type in df_total.type.unique():
            for cfg in df_total.cfg.unique():
                dfx = df_total[(df_total.cfg == cfg) & (df_total.type == type)]
                dfx_networks = [x for x in dfx.network.unique()]
                if all([x in dfx_networks for x in ['mobilenet_v1','deepautoencoder','resnet8','ds_cnn']]):
                    geom_mean = 1
                    weighted_geom_mean = 0
                    total_MAC_count = 0
                    for i,r in dfx.iterrows():
                        if r.network not in ['geom_mean','weighted_geom_mean']:
                            geom_mean *= r[parameter]
                            weighted_geom_mean += np.log(r[parameter]) * r.total_MAC_count
                            total_MAC_count += r.total_MAC_count
                    geom_mean = np.power(geom_mean, 1/4)
                    weighted_geom_mean = np.exp(weighted_geom_mean/total_MAC_count)
                    if 'geom_mean' not in dfx_networks:
                        r_gm = deepcopy(dfx.iloc[0])
                        r_wgm = deepcopy(dfx.iloc[0])
                        r_gm.network = 'geom_mean'
                        r_gm[parameter] = geom_mean
                        if type.split(' ')[0] == 'dram_weights':
                            r_gm['area'] = r_gm.area - r_gm.weight_sram_area - r_gm.act_sram_area + 0.157
                            r_wgm['area'] = r_wgm.area - r_wgm.weight_sram_area - r_wgm.act_sram_area + 0.157
                        else:
                            r_gm['area'] = r_gm.area - r_gm.weight_sram_area - r_gm.act_sram_area + 0.157+0.8
                            r_wgm['area'] = r_wgm.area - r_wgm.weight_sram_area - r_wgm.act_sram_area + 0.157+0.8

                        r_wgm.network = 'weighted_geom_mean'
                        r_wgm[parameter] = weighted_geom_mean
                        df_total = df_total.append(r_gm)#, ignore_index=True)
                        df_total = df_total.append(r_wgm)#, ignore_index=True)
                    else:
                        df_total.loc[(df_total.type == type) & (df_total.cfg == cfg) & (df_total.network == 'geom_mean'), parameter] = geom_mean
                        df_total.loc[(df_total.type == type) & (df_total.cfg == cfg) & (df_total.network == 'weighted_geom_mean'), parameter] = weighted_geom_mean


            df_total = df_total.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)
            for  netw_gm in ['geom_mean','weighted_geom_mean']:
                dfx = df_total[(df_total.type == type) & (df_total.network == netw_gm)]
                df_gm_pareto = pd.DataFrame()
                best_gm = float('inf')
                for i,r in dfx.iterrows():
                    if r[parameter] < best_gm:
                        df_gm_pareto = df_gm_pareto.append(r)
                        best_gm = r[parameter]
                df_gm_pareto['parameter'] = parameter
                df_gm_pareto['type'] = type
                df_gm_pareto['network'] = netw_gm
                df_pareto_total = pd.concat([df_pareto_total, df_gm_pareto])


        lines = {0:'-',1:':',2:'--',3:'-.'}
#        colors= plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig = make_subplots(4,1,subplot_titles=[x for x in df_total.network.unique()])
        df_pareto_total_tmp = df_pareto_total.copy(deep=True)
        df_pareto_total = df_pareto_total.sort_values(by=['area'],ascending=[True],ignore_index=True)
        df_pareto_total = df_pareto_total[(df_pareto_total.network != 'geom_mean') & (df_pareto_total.network != 'weighted_geom_mean')]
        for ii_network, network in enumerate(df_pareto_total.network.unique()):
            ax[ii_network].set_title(network)
            ax[ii_network].grid()
            ax[ii_network].set_xlabel('Area [$\mu$m$^2$]')
            if ii_network == 0:
                ax[ii_network].set_ylabel(parameter)
            ax[ii_network].axvline(x=area_net[network]['act'],color='#c0beb3',lw=4,ls='--')
            ax[ii_network].axvline(x=area_net[network]['weight']+area_net[network]['act'],color=colors[1],lw=4,ls='--')
            for type in df_pareto_total.type.unique():
                if type.split(' ')[0] == 'dram_weights':
                    ii_c = 0
                elif type.split(' ')[0] == 'weights_on_chip':
                    ii_c = 1
                else:
                    ii_c = 2
                if type.split(' ')[1] == 'no_d3':
                    ls = 0
                else:
                    ls = 1
                
                dfx =  df_total[(df_total.network == network) & (df_total.type == type)]
                dfx_pareto =  df_pareto_total[(df_pareto_total.network == network) & (df_pareto_total.type == type)]
                ax[ii_network].scatter(dfx['area'],dfx[parameter],label=type,c=colors[ii_c],alpha=0.1)
                if ii_c == 1:
                    lw = 10
                else:
                    lw = 3
                ax[ii_network].plot(dfx_pareto['area'],dfx_pareto[parameter],c=colors[ii_c],ls=lines[ls],lw=lw, label=type.split(' ')[0])
#                for i,r in dfx_pareto.iterrows():
#                    if i%2 == 0:
#                        ax[ii_network].text(r['area'],r[parameter],str(int(r['M'])), color='black',path_effects=[pe.withStroke(linewidth=4, foreground='w')])#bbox={'facecolor':colors[ii_c],'pad':1,'edgecolor':None})
                ax[ii_network].set_yscale('log')
                ax[ii_network].set_xscale('log')
            #    fig.add_trace(go.Scatter(x = dfx['area'],y=dfx[parameter],mode='markers',name=f'{network} {type}'), col=(ii_network%2)+1, row=(ii_network//2)+1)
            #    fig.add_trace(go.Scatter(x = dfx_pareto['area'],y=dfx_pareto[parameter],mode='markers+lines',name=f'{network} {type} pareto'), col=(ii_network%2)+1, row=(ii_network//2)+1)
#            ax[ii_network].legend()
            ax[ii_network].set_xlim(right=10)
        plt.tight_layout()
        plt.show()
        #breakpoint()
        #fig.update_xaxes(type='log')
        #fig.update_yaxes(type='log')
        #fig.show()
        dfx_pareto = df_pareto_total_tmp[(df_pareto_total_tmp.network == 'geom_mean') & (df_pareto_total_tmp.parameter == parameter)]
        dfx = df_total[(df_total.network == 'geom_mean') & (df_total.parameter == parameter)]
        plt.axvline(x=area_net['mobilenet_v1']['act'],color=colors[0],lw=4)
        plt.axvline(x=area_net['deepautoencoder']['weight']+area_net[network]['act'],color=color[1],lw=4)


        for c in dfx_pareto.type.unique():
            if c.split(' ')[0] == 'dram_weights':
                ii_c = 0
            elif c.split(' ')[0] == 'weights_on_chip':
                ii_c = 1
            else:
                ii_c = 2
            if c.split(' ')[1] == 'no_d3':
                ls = 0
            else:
                ls = 1
 
        #    dfxx =dfx[dfx.type == c]
            dfxx_pareto =dfx_pareto[dfx_pareto.type == c]
            plt.plot(dfxx_pareto['area'],dfxx_pareto[parameter],c=colors[ii_c],ls=lines[ls],lw=4,marker='o',markersize=3,label=c)
        plt.xlabel('Area [$\mu$m$^2$]')
        plt.ylabel(parameter)
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.grid()
        plt.title('Geom mean')
        plt.show()


def fig_pareto_plot_d1d2d3m(parameter=None):
    area_net = {'mobilenet_v1':{'act':0.157,'weight':0.59},
            'resnet8':{'act':0.093, 'weight':0.213},
            'deepautoencoder':{'act':0.0058, 'weight':0.8},
            'ds_cnn':{'act':0.046, 'weight':0.066},
            'geom_mean':{'act':0.147, 'weight':0.213},
            'weighted_geom_mean':{'act':0.147, 'weight':0.213}}
            
    imc_type = 'dimc'

    colors = ['#ff7e0e', '#5ba4bf', '#afe9af','#912f56']
    #for parameter in ['latency_total','energy_total','EDP']:
    for parameter in ['EDP']:
        df_total = pd.DataFrame()
        df_pareto_total = pd.DataFrame()
        fig,ax = plt.subplots(1,4, figsize=(7,5))
        for no_d3 in [True,False]:
            for network in ['ds_cnn','resnet8','deepautoencoder','mobilenet_v1']:
#            for network in ['mobilenet_v1']:
                if False:#network in ['ds_cnn','resnet8','mobilenet_v1']:
                    with open(f"outputs_{network}/{imc_type}/var/imc_weights_{network}.pkl","rb") as infile:
                        df_all_weights = pickle.load(infile)

                    df_all_weights['EDP'] = df_all_weights['latency_total'] * df_all_weights['energy_total']
                    df_all_weights['type'] = 'all_weights_on_chip'
                    df_all_weights = df_all_weights.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)
                    df_aw_pareto = pd.DataFrame()
                    best_latency = float('inf')
                    for i,r in df_all_weights.iterrows():
                        if r[parameter] < best_latency:
                            df_aw_pareto = df_aw_pareto.append(r)
                            best_latency = r[parameter]
                    df_aw_pareto['type'] = 'all_weight_pareto'
                else:
                    df_aw_pareto = pd.DataFrame()
                    df_all_weights = pd.DataFrame()

                with open(f"outputs_{network}/{imc_type}/var/onchip_weights_{network}.pkl","rb") as infile:
                    df_onchip_weights = pickle.load(infile)
                df_onchip_weights['EDP'] = df_onchip_weights['latency_total'] * df_onchip_weights['energy_total']
                if no_d3:
                    df_onchip_weights = df_onchip_weights[df_onchip_weights.D3 == 1]
                    df_onchip_weights['type'] = 'weights_on_chip no_d3'
                else:
                    df_onchip_weights = df_onchip_weights[df_onchip_weights.D3 != 1]
                    df_onchip_weights['type'] = 'weights_on_chip d3'
                df_onchip_weights = df_onchip_weights.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)
                df_onchip_pareto = pd.DataFrame()
                best_latency = float('inf')
                for i,r in df_onchip_weights.iterrows():
                    if r[parameter] < best_latency:
                        df_onchip_pareto = df_onchip_pareto.append(r)
                        best_latency = r[parameter]
                if no_d3:
                    df_onchip_pareto['type'] = 'weights_on_chip no_d3'
                else:
                    df_onchip_pareto['type'] = 'weights_on_chip d3'
                df_onchip_pareto = df_onchip_pareto.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)

                with open(f"outputs_{network}/{imc_type}/var/offchip_weights_{network}.pkl","rb") as infile:
                    df_offchip_weights = pickle.load(infile)
                df_offchip_weights['EDP'] = df_offchip_weights['latency_total'] * df_offchip_weights['energy_total']
                if no_d3:
                    df_offchip_weights = df_offchip_weights[df_offchip_weights.D3 == 1]
                    df_offchip_weights['type'] = 'dram_weights no_d3'
                else:
                    df_offchip_weights = df_offchip_weights[df_offchip_weights.D3 != 1]
                    df_offchip_weights['type'] = 'dram_weights d3'

                df_offchip_weights = df_offchip_weights.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)
                df_offchip_pareto = pd.DataFrame()
                best_latency = float('inf')
                for i,r in df_offchip_weights.iterrows():
                    if r[parameter] < best_latency:
                        df_offchip_pareto = df_offchip_pareto.append(r)
                        best_latency = r[parameter]
                if no_d3:
                    df_offchip_pareto['type'] = 'dram_weights no_d3'
                else:
                    df_offchip_pareto['type'] = 'dram_weights d3'
                df_offchip_pareto = df_offchip_pareto.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)

                df = pd.concat([df_onchip_weights, df_offchip_weights, df_all_weights])
                df_pareto = pd.concat([df_onchip_pareto, df_offchip_pareto, df_aw_pareto])
                df['network'] = network
                df_pareto['network'] = network
                df_pareto['parameter'] = parameter        
                df_total['parameter'] = parameter
                if df_total.empty:
                    df_total = df.copy()
                else:
                    df_total = pd.concat([df, df_total])
                
                if df_pareto_total.empty:
                    df_pareto_total = df_pareto.copy()
                else:
                    df_pareto_total = pd.concat([df_pareto, df_pareto_total])
            with open(f'outputs_{network}/{imc_type}/var/tinymlperf_pareto_{parameter}.pkl','wb') as infile:
                pickle.dump(df_pareto_total,infile)
        

        for type in df_total.type.unique():
            for cfg in df_total.cfg.unique():
                dfx = df_total[(df_total.cfg == cfg) & (df_total.type == type)]
                dfx_networks = [x for x in dfx.network.unique()]
                if all([x in dfx_networks for x in ['mobilenet_v1','deepautoencoder','resnet8','ds_cnn']]):
                    geom_mean = 1
                    weighted_geom_mean = 0
                    total_MAC_count = 0
                    for i,r in dfx.iterrows():
                        if r.network not in ['geom_mean','weighted_geom_mean']:
                            geom_mean *= r[parameter]
                            weighted_geom_mean += np.log(r[parameter]) * r.total_MAC_count
                            total_MAC_count += r.total_MAC_count
                    geom_mean = np.power(geom_mean, 1/4)
                    weighted_geom_mean = np.exp(weighted_geom_mean/total_MAC_count)
                    if 'geom_mean' not in dfx_networks:
                        r_gm = deepcopy(dfx.iloc[0])
                        r_wgm = deepcopy(dfx.iloc[0])
                        r_gm.network = 'geom_mean'
                        r_gm[parameter] = geom_mean
                        if type.split(' ')[0] == 'dram_weights':
                            r_gm['area'] = r_gm.area - r_gm.weight_sram_area - r_gm.act_sram_area + 0.157
                            r_wgm['area'] = r_wgm.area - r_wgm.weight_sram_area - r_wgm.act_sram_area + 0.157
                        else:
                            r_gm['area'] = r_gm.area - r_gm.weight_sram_area - r_gm.act_sram_area + 0.157+0.8
                            r_wgm['area'] = r_wgm.area - r_wgm.weight_sram_area - r_wgm.act_sram_area + 0.157+0.8

                        r_wgm.network = 'weighted_geom_mean'
                        r_wgm[parameter] = weighted_geom_mean
                        df_total = df_total.append(r_gm)#, ignore_index=True)
                        df_total = df_total.append(r_wgm)#, ignore_index=True)
                    else:
                        df_total.loc[(df_total.type == type) & (df_total.cfg == cfg) & (df_total.network == 'geom_mean'), parameter] = geom_mean
                        df_total.loc[(df_total.type == type) & (df_total.cfg == cfg) & (df_total.network == 'weighted_geom_mean'), parameter] = weighted_geom_mean


            df_total = df_total.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)
            for  netw_gm in ['geom_mean','weighted_geom_mean']:
                dfx = df_total[(df_total.type == type) & (df_total.network == netw_gm)]
                df_gm_pareto = pd.DataFrame()
                best_gm = float('inf')
                for i,r in dfx.iterrows():
                    if r[parameter] < best_gm:
                        df_gm_pareto = df_gm_pareto.append(r)
                        best_gm = r[parameter]
                df_gm_pareto['parameter'] = parameter
                df_gm_pareto['type'] = type
                df_gm_pareto['network'] = netw_gm
                df_pareto_total = pd.concat([df_pareto_total, df_gm_pareto])

#        dfx = df_total[(df_total.type == 'weights_on_chip d3') | (df_total.type == 'weights_on_chip no_d3')]
#        dfx['M_str'] = dfx['M'].astype(str)
#        dfx['D3_str'] = dfx['D3'].astype(str)
#        dfx = dfx.sort_values(by=['area','EDP'],ascending=[True,True],ignore_index=True)
#        df_d3_pareto = pd.DataFrame()
#        for network in dfx.network.unique():
#            for d3 in dfx.D3.unique():
#                dfxx = dfx[(dfx.D3 == d3) & (dfx.network == network)]
#                best_edp = float('inf')
#                for i,r in dfxx.iterrows():
#                    if r.EDP < best_edp:
#                        df_d3_pareto = df_d3_pareto.append(r)
#                        best_edp = r.EDP
#
#        fig = px.scatter(df_d3_pareto,'area','EDP',symbol='M_str',color='D3_str',log_x=True,log_y=True,facet_col='D3',facet_row='network')
#        fig.show()
#        breakpoint()

        lines = {0:'-',1:':',2:'--',3:'-.'}
        fig = make_subplots(2,2,subplot_titles=[x for x in df_total.network.unique()])
        df_pareto_total_tmp = df_pareto_total.copy(deep=True)
        df_pareto_total = df_pareto_total.sort_values(by=['area'],ascending=[True],ignore_index=True)
        df_pareto_total = df_pareto_total[(df_pareto_total.network != 'geom_mean') & (df_pareto_total.network != 'weighted_geom_mean')]
        for ii_network, network in enumerate(df_pareto_total.network.unique()):
            ax[ii_network].set_title(network)
            ax[ii_network].grid()
            ax[ii_network].set_xlabel('Area [$\mu$m$^2$]')
            if ii_network == 0:
                ax[ii_network].set_ylabel('EDP')

#            ax[ii_network].axvline(x=area_net[network]['act'],color='tomato',lw=4)
            ax[ii_network].axvline(x=area_net[network]['weight']+area_net[network]['act'],color=colors[1],ls='--',lw=4)
            for type in ['weights_on_chip no_d3','weights_on_chip d3']:
                if type.split(' ')[0] == 'dram_weights':
                    ii_c = 0
                elif type.split(' ')[0] == 'weights_on_chip':
                    ii_c = 1
                else:
                    ii_c = 2
                if type.split(' ')[1] == 'no_d3':
                    ls = 1
                else:
                    ls = 0
                
                dfx =  df_total[(df_total.network == network) & (df_total.type == type)]
                dfx_pareto =  df_pareto_total[(df_pareto_total.network == network) & (df_pareto_total.type == type)]
                if ls == 1:
                    ax[ii_network].scatter(dfx['area'],dfx[parameter],label=type,c=colors[1],alpha=0.1,marker='o')
                    ax[ii_network].plot(dfx_pareto['area'],dfx_pareto[parameter],c=colors[1], lw=3, label=type.split(' ')[0])
                else:
                    ax[ii_network].scatter(dfx['area'],dfx[parameter],label=type,c=colors[3],alpha=0.03,marker='o')
                    ax[ii_network].plot(dfx_pareto['area'],dfx_pareto[parameter],c=colors[3],lw=3, label=type.split(' ')[0])

                #for i,r in dfx_pareto.iterrows():
                #    if r['area'] < 5:
                #        if ls == 1:
                #            ax[ii_network].text(r['area'],r[parameter],f'{str(int(r.M))}', color='white',path_effects=[pe.withStroke(linewidth=4, foreground='orange')])#bbox={'facecolor':colors[ii_c],'pad':1,'edgecolor':None})
                #        else:
                #            #ax[ii_network].text(r['area'],r[parameter],str(int(r['M'])), color='white',path_effects=[pe.withStroke(linewidth=4, foreground='purple')])#bbox={'facecolor':colors[ii_c],'pad':1,'edgecolor':None})
                #            ax[ii_network].text(r['area'],r[parameter],f'{str(int(r.M))}', color='white',path_effects=[pe.withStroke(linewidth=4, foreground='purple')])#bbox={'facecolor':colors[ii_c],'pad':1,'edgecolor':None})
                ax[ii_network].set_yscale('log')
                ax[ii_network].set_xscale('log')
            #    fig.add_trace(go.Scatter(x = dfx['area'],y=dfx[parameter],mode='markers',name=f'{network} {type}'), col=(ii_network%2)+1, row=(ii_network//2)+1)
            #    fig.add_trace(go.Scatter(x = dfx_pareto['area'],y=dfx_pareto[parameter],mode='markers+lines',name=f'{network} {type} pareto'), col=(ii_network%2)+1, row=(ii_network//2)+1)
#            ax[ii_network].legend()
#            ax[ii_network].set_xlim(right=5)
        plt.tight_layout()
        plt.show()
        #breakpoint()
        #fig.update_xaxes(type='log')
        #fig.update_yaxes(type='log')
        #fig.show()
        with open('pareto_points.pkl','wb') as infile:
            pickle.dump(df_pareto_total, infile)

        dfx_pareto = df_pareto_total_tmp[(df_pareto_total_tmp.network == 'geom_mean') & (df_pareto_total_tmp.parameter == parameter)]
        dfx = df_total[(df_total.network == 'geom_mean') & (df_total.parameter == parameter)]
        plt.axvline(x=area_net['mobilenet_v1']['act'],color='tomato',lw=4)
        plt.axvline(x=area_net['deepautoencoder']['weight']+area_net[network]['act'],color='darkkhaki',lw=4)


        for c in dfx_pareto.type.unique():
            if c.split(' ')[0] == 'dram_weights':
                ii_c = 0
            elif c.split(' ')[0] == 'weights_on_chip':
                ii_c = 1
            else:
                ii_c = 2
            if c.split(' ')[1] == 'no_d3':
                ls = 0
            else:
                ls = 1
 
        #    dfxx =dfx[dfx.type == c]
            dfxx_pareto =dfx_pareto[dfx_pareto.type == c]
            plt.plot(dfxx_pareto['area'],dfxx_pareto[parameter],c=colors[ii_c],ls=lines[ls],lw=4,marker='o',markersize=3,label=c)
        plt.xlabel('Area [$\mu$m$^2$]')
        plt.ylabel(parameter)
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.grid()
        plt.title('Geom mean')
        plt.show()


def fig_pareto_plot_d1d2d3m_memory_shrink(parameter=None):
    area_net = {'mobilenet_v1':{'act':0.157,'weight':0.59},
            'resnet8':{'act':0.093, 'weight':0.213},
            'deepautoencoder':{'act':0.0058, 'weight':0.8},
            'ds_cnn':{'act':0.046, 'weight':0.066},
            'geom_mean':{'act':0.147, 'weight':0.213},
            'weighted_geom_mean':{'act':0.147, 'weight':0.213}}

    with open('cacti_table.pkl','rb') as infile:
        mem_data = pickle.load(infile)
    with open('pareto_points.pkl','rb') as infile:
        pareto_data = pickle.load(infile)

    # RESNET8
    mem_data['sram_w_cost'] /= 128
    mem_req = {'resnet8':{'W': 77360,'A': 32768},
            'deepautoencoder':{'W':264192,'A': 768 + 1024},
            'ds_cnn':{'W': 22016, 'A': 16000},
            'mobilenet_v1':{'W': 208112,'A': 55296}}
    
    df = pd.DataFrame()
    imc_type ='dimc'
    for network in ['resnet8','ds_cnn','mobilenet_v1']:
        dfn = get_df(f"./outputs_{network}/{imc_type}/", network)
        dfn = dfn[dfn.cfg == dfn.iloc[0]['cfg']]
        layer_size = []
        for i,r in dfn.iterrows():
            size = 1
            for loop in ['C','OX','OY']:
                if loop in r.loop_dim_size:
                    size *= r.loop_dim_size[loop]
            layer_size.append(size)
        pdx = pareto_data[(pareto_data.network == network) & ((pareto_data.type == 'weights_on_chip d3') | (pareto_data.type == 'weights_on_chip no_d3'))]
        weight_req = mem_req[network]['W']
        act_req = mem_req[network]['A']
        mdw = mem_data[mem_data.sram_size <= weight_req]
        mda = mem_data[mem_data.sram_size <= act_req]
        original_weight_cost = mdw.iloc[-1].sram_w_cost
        original_act_cost = mda.iloc[-1].sram_w_cost
        for i,r in pdx.iterrows():
            for i_mem, r_mem in mdw.iterrows():
                rx = r.copy(deep=True)
                rx.area = rx.area - rx.weight_sram_area + r_mem.sram_area
                rx['weight_sram_size'] = r_mem.sram_size
                rx['sram_weight_area'] = r_mem.sram_area
                weight_req_4kb = (weight_req - r_mem.sram_size)*8
                rx.energy_total = rx.energy_total + (3.7 * weight_req_4kb - (original_weight_cost - r_mem.sram_w_cost) * weight_req * 8)
                rx.latency_total = rx.latency_total + (weight_req_4kb //32)
                rx.EDP = rx.energy_total * rx.latency_total
                rx.type = 'w_mem_shrink'
                df = df.append(rx)
                ra = rx.copy(deep=True)
                for i_amem, ra_mem in mda.iterrows():
                    ra.area = ra.area - ra.act_sram_area + ra_mem.sram_area
                    ra['act_sram_size'] = ra_mem.sram_size
                    ra['sram_act_area'] = ra_mem.sram_area
                    extra_mem = 0
                    for l in layer_size:
                        if l > ra_mem.sram_size:
                            extra_mem += (l - ra_mem.sram_size)
                    extra_mem *= 8
                    ra.energy_total = ra.energy_total + (2 * 3.7 * extra_mem - (original_act_cost - ra_mem.sram_w_cost) * sum(layer_size)*2 * 8)
                    ra.latency_total = ra.latency_total + (extra_mem //32)
                    ra.EDP = ra.energy_total * ra.latency_total
                    ra.type = 'a+w_mem_shrink'
                    df = df.append(ra)


        pd_total = pd.concat([df, pareto_data[pareto_data.network == network]])
        fig = px.scatter(pd_total,'area','EDP',color='type',log_x=True,log_y=True,hover_data=['cfg'])
        fig.show()
        fig = px.scatter(pd_total,'area','latency_total',color='type',log_x=True,log_y=True,hover_data=['cfg'])
        fig.show()
        fig = px.scatter(pd_total,'area','energy_total',color='type',log_x=True,log_y=True,hover_data=['cfg'])
        fig.show()


        breakpoint()
    lines = {0:'-',1:':',2:'--',3:'-.'}
    colors= plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig = make_subplots(2,2,subplot_titles=[x for x in df_total.network.unique()])
    df_pareto_total_tmp = df_pareto_total.copy(deep=True)
    df_pareto_total = df_pareto_total.sort_values(by=['area'],ascending=[True],ignore_index=True)
    df_pareto_total = df_pareto_total[(df_pareto_total.network != 'geom_mean') & (df_pareto_total.network != 'weighted_geom_mean')]
    for ii_network, network in enumerate(df_pareto_total.network.unique()):
        ax[ii_network].set_title(network)
        ax[ii_network].grid()
        ax[ii_network].set_xlabel('Area [$\mu$m$^2$]')
        if ii_network == 0:
            ax[ii_network].set_ylabel(parameter)
#            ax[ii_network].axvline(x=area_net[network]['act'],color='tomato',lw=4)
        ax[ii_network].axvline(x=area_net[network]['weight']+area_net[network]['act'],color='darkkhaki',lw=4)
        for type in ['weights_on_chip no_d3','weights_on_chip d3']:
            if type.split(' ')[0] == 'dram_weights':
                ii_c = 0
            elif type.split(' ')[0] == 'weights_on_chip':
                ii_c = 1
            else:
                ii_c = 2
            if type.split(' ')[1] == 'no_d3':
                ls = 1
            else:
                ls = 0
            
            dfx =  df_total[(df_total.network == network) & (df_total.type == type)]
            dfx_pareto =  df_pareto_total[(df_pareto_total.network == network) & (df_pareto_total.type == type)]
            if ls == 1:
                ax[ii_network].scatter(dfx['area'],dfx[parameter],label=type,c='orange',alpha=0.4,marker='o')
                ax[ii_network].plot(dfx_pareto['area'],dfx_pareto[parameter],c='orange',ls=lines[ls],lw=3,marker='o',markersize=3, label=type.split(' ')[0])
            else:
                ax[ii_network].scatter(dfx['area'],dfx[parameter],label=type,c='purple',alpha=0.1,marker='^')
                ax[ii_network].plot(dfx_pareto['area'],dfx_pareto[parameter],c='purple',ls=lines[ls],lw=3,marker='o',markersize=3, label=type.split(' ')[0])

#                for i,r in dfx_pareto.iterrows():
#                    if i%2 == 0:
#                        ax[ii_network].text(r['area'],r[parameter],str(int(r['M'])), color='black',path_effects=[pe.withStroke(linewidth=4, foreground='w')])#bbox={'facecolor':colors[ii_c],'pad':1,'edgecolor':None})
            ax[ii_network].set_yscale('log')
            ax[ii_network].set_xscale('log')
        #    fig.add_trace(go.Scatter(x = dfx['area'],y=dfx[parameter],mode='markers',name=f'{network} {type}'), col=(ii_network%2)+1, row=(ii_network//2)+1)
        #    fig.add_trace(go.Scatter(x = dfx_pareto['area'],y=dfx_pareto[parameter],mode='markers+lines',name=f'{network} {type} pareto'), col=(ii_network%2)+1, row=(ii_network//2)+1)
#            ax[ii_network].legend()
       # ax[ii_network].set_xlim(right=20)
    plt.tight_layout()
    plt.show()
    #breakpoint()
    #fig.update_xaxes(type='log')
    #fig.update_yaxes(type='log')
    #fig.show()
    dfx_pareto = df_pareto_total_tmp[(df_pareto_total_tmp.network == 'geom_mean') & (df_pareto_total_tmp.parameter == parameter)]
    dfx = df_total[(df_total.network == 'geom_mean') & (df_total.parameter == parameter)]
    plt.axvline(x=area_net['mobilenet_v1']['act'],color='tomato',lw=4)
    plt.axvline(x=area_net['deepautoencoder']['weight']+area_net[network]['act'],color='darkkhaki',lw=4)


    for c in dfx_pareto.type.unique():
        if c.split(' ')[0] == 'dram_weights':
            ii_c = 0
        elif c.split(' ')[0] == 'weights_on_chip':
            ii_c = 1
        else:
            ii_c = 2
        if c.split(' ')[1] == 'no_d3':
            ls = 0
        else:
            ls = 1

    #    dfxx =dfx[dfx.type == c]
        dfxx_pareto =dfx_pareto[dfx_pareto.type == c]
        plt.plot(dfxx_pareto['area'],dfxx_pareto[parameter],c=colors[ii_c],ls=lines[ls],lw=4,marker='o',markersize=3,label=c)
    plt.xlabel('Area [$\mu$m$^2$]')
    plt.ylabel(parameter)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.title('Geom mean')
    plt.show()


def fig_plot_utilization_d1d2m():
    area_net = {'mobilenet_v1':{'act':0.157,'weight':0.59},
            'resnet8':{'act':0.093, 'weight':0.213},
            'deepautoencoder':{'act':0.0058, 'weight':0.8},
            'ds_cnn':{'act':0.046, 'weight':0.066}}
            
    colors = ["#51a3a3","#75485e","#cb904d","#af9ab2","#cbe896"]
    imc_type = 'dimc'
    for parameter in ['spat_utilization','temp_utilization','utilization']:
        df_total = pd.DataFrame()
        df_pareto_total = pd.DataFrame()
        for no_d3 in [True]:
            for network in ['ds_cnn','resnet8','deepautoencoder','mobilenet_v1']:
#            for network in ['mobilenet_v1','ds_cnn','deepautoencoder']:
                with open(f"outputs_{network}/{imc_type}/var/onchip_weights_{network}.pkl","rb") as infile:
                    df_onchip_weights = pickle.load(infile)
                df_onchip_weights['utilization_norm'] = df_onchip_weights[parameter] / df_onchip_weights['total_MAC_count']
                if no_d3:
                    df_onchip_weights = df_onchip_weights[df_onchip_weights.D3 == 1]
                    df_onchip_weights['type'] = 'weights_on_chip no_d3'
                else:
                    df_onchip_weights = df_onchip_weights[df_onchip_weights.D3 != 1]
                    df_onchip_weights['type'] = 'weights_on_chip d3'
                df_onchip_weights = df_onchip_weights.sort_values(by=['area',parameter],ascending=[False,True],ignore_index=True)
                df_onchip_pareto = pd.DataFrame()
                best_latency = 0
                for i,r in df_onchip_weights.iterrows():
                    if r[parameter] > best_latency:
                        df_onchip_pareto = df_onchip_pareto.append(r)
                        best_latency = r[parameter]
                if no_d3:
                    df_onchip_pareto['type'] = 'weights_on_chip no_d3'
                else:
                    df_onchip_pareto['type'] = 'weights_on_chip d3'
                df_onchip_pareto = df_onchip_pareto.sort_values(by=['area',parameter],ascending=[False,True],ignore_index=True)


                with open(f"outputs_{network}/{imc_type}/var/offchip_weights_{network}.pkl","rb") as infile:
                    df_offchip_weights = pickle.load(infile)
                df_offchip_weights['utilization_norm'] = df_offchip_weights[parameter] / df_offchip_weights['total_MAC_count']
                if no_d3:
                    df_offchip_weights = df_offchip_weights[df_offchip_weights.D3 == 1]
                    df_offchip_weights['type'] = 'dram_weights no_d3'
                else:
                    df_offchip_weights = df_offchip_weights[df_offchip_weights.D3 != 1]
                    df_offchip_weights['type'] = 'dram_weights d3'

                df_offchip_weights = df_offchip_weights.sort_values(by=['area',parameter],ascending=[False,True],ignore_index=True)
                df_offchip_pareto = pd.DataFrame()
                best_latency = 0
                for i,r in df_offchip_weights.iterrows():
                    if r[parameter] > best_latency:
                        df_offchip_pareto = df_offchip_pareto.append(r)
                        best_latency = r[parameter]
                if no_d3:
                    df_offchip_pareto['type'] = 'dram_weights no_d3'
                else:
                    df_offchip_pareto['type'] = 'dram_weights d3'
                df_offchip_pareto = df_offchip_pareto.sort_values(by=['area',parameter],ascending=[False,True],ignore_index=True)

                df = pd.concat([df_onchip_weights, df_offchip_weights])
                df_pareto = pd.concat([df_onchip_pareto, df_offchip_pareto])
                df['network'] = network
                df_pareto['network'] = network
                df_pareto['parameter'] = parameter        

                if df_total.empty:
                    df_total = df.copy()
                else:
                    df_total = pd.concat([df, df_total])
                
                if df_pareto_total.empty:
                    df_pareto_total = df_pareto.copy()
                else:
                    df_pareto_total = pd.concat([df_pareto, df_pareto_total])
            with open(f'outputs_{network}/{imc_type}/var/tinymlperf_pareto_{parameter}.pkl','wb') as infile:
                pickle.dump(df_pareto_total,infile)
#        dfx = df_total[df_total.type == 'dram_weights no_d3']
#        df_total['utilization_norm'] = df_total['utilization'] / df_total['total_MAC_count']

        if parameter == 'utilization':
            plt.subplot(1,2,1)
        if parameter == 'spat_utilization':
            plt.subplot(2,2,2)
        if parameter == 'temp_utilization':
            plt.subplot(2,2,4)

        lines = {0:'-',1:':',2:'--',3:'-.'}
        for ii_network, network in enumerate(df_pareto_total.network.unique()):
            for ii_c, c in enumerate(df_pareto_total.type.unique()):
                if c.split(' ')[0] == 'weights_on_chip':
                    d3 = c.split(' ')[1]
                    dfx_pareto = df_pareto_total[(df_pareto_total.network == network) & (df_pareto_total.type == c)]
                    dfx= df_total[(df_total.network == network) & (df_total.type == c)]
                    plt.scatter(dfx['area'],dfx['utilization_norm'], c=colors[ii_network], ls=lines[ii_c],  alpha=0.05)
                    plt.plot(dfx_pareto['area'],dfx_pareto['utilization_norm'], c=colors[ii_network], ls=lines[ii_c], label=f'{network}',lw=3,marker='o',markersize=5)

        plt.xscale('log')
        plt.ylabel(f'{parameter} [%]')
        if parameter != 'spat_utilization':
            plt.xlabel('Area [$\mu$m$^2$]')
        if parameter == 'utilization':
            plt.legend(loc='lower left')
        plt.grid()
    plt.tight_layout()
    plt.show()
        
def fig_plot_utilization_d1d2d3m():
    area_net = {'mobilenet_v1':{'act':0.157,'weight':0.59},
            'resnet8':{'act':0.093, 'weight':0.213},
            'deepautoencoder':{'act':0.0058, 'weight':0.8},
            'ds_cnn':{'act':0.046, 'weight':0.066}}
            
    imc_type = 'dimc'
    for parameter in ['spat_utilization','temp_utilization','utilization']:
        df_total = pd.DataFrame()
        df_pareto_total = pd.DataFrame()
        for no_d3 in [True,False]:
            for network in ['ds_cnn','resnet8','deepautoencoder','mobilenet_v1']:
#            for network in ['mobilenet_v1','ds_cnn','deepautoencoder']:
                with open(f"outputs_{network}/{imc_type}/var/onchip_weights_{network}.pkl","rb") as infile:
                    df_onchip_weights = pickle.load(infile)
                df_onchip_weights['utilization_norm'] = df_onchip_weights[parameter] / df_onchip_weights['total_MAC_count']
                if no_d3:
                    df_onchip_weights = df_onchip_weights[df_onchip_weights.D3 == 1]
                    df_onchip_weights['type'] = 'weights_on_chip no_d3'
                else:
                    df_onchip_weights = df_onchip_weights[df_onchip_weights.D3 != 1]
                    df_onchip_weights['type'] = 'weights_on_chip d3'
                df_onchip_weights = df_onchip_weights.sort_values(by=['area',parameter],ascending=[False,True],ignore_index=True)
                df_onchip_pareto = pd.DataFrame()
                best_latency = 0
                for i,r in df_onchip_weights.iterrows():
                    if r[parameter] > best_latency:
                        df_onchip_pareto = df_onchip_pareto.append(r)
                        best_latency = r[parameter]
                if no_d3:
                    df_onchip_pareto['type'] = 'weights_on_chip no_d3'
                else:
                    df_onchip_pareto['type'] = 'weights_on_chip d3'
                df_onchip_pareto = df_onchip_pareto.sort_values(by=['area',parameter],ascending=[False,True],ignore_index=True)


                with open(f"outputs_{network}/{imc_type}/var/offchip_weights_{network}.pkl","rb") as infile:
                    df_offchip_weights = pickle.load(infile)
                df_offchip_weights['utilization_norm'] = df_offchip_weights[parameter] / df_offchip_weights['total_MAC_count']
                if no_d3:
                    df_offchip_weights = df_offchip_weights[df_offchip_weights.D3 == 1]
                    df_offchip_weights['type'] = 'dram_weights no_d3'
                else:
                    df_offchip_weights = df_offchip_weights[df_offchip_weights.D3 != 1]
                    df_offchip_weights['type'] = 'dram_weights d3'

                df_offchip_weights = df_offchip_weights.sort_values(by=['area',parameter],ascending=[False,True],ignore_index=True)
                df_offchip_pareto = pd.DataFrame()
                best_latency = 0
                for i,r in df_offchip_weights.iterrows():
                    if r[parameter] > best_latency:
                        df_offchip_pareto = df_offchip_pareto.append(r)
                        best_latency = r[parameter]
                if no_d3:
                    df_offchip_pareto['type'] = 'dram_weights no_d3'
                else:
                    df_offchip_pareto['type'] = 'dram_weights d3'
                df_offchip_pareto = df_offchip_pareto.sort_values(by=['area',parameter],ascending=[False,True],ignore_index=True)

                df = pd.concat([df_onchip_weights, df_offchip_weights])
                df_pareto = pd.concat([df_onchip_pareto, df_offchip_pareto])
                df['network'] = network
                df_pareto['network'] = network
                df_pareto['parameter'] = parameter        

                if df_total.empty:
                    df_total = df.copy()
                else:
                    df_total = pd.concat([df, df_total])
                
                if df_pareto_total.empty:
                    df_pareto_total = df_pareto.copy()
                else:
                    df_pareto_total = pd.concat([df_pareto, df_pareto_total])
            with open(f'outputs_{network}/{imc_type}/var/tinymlperf_pareto_{parameter}.pkl','wb') as infile:
                pickle.dump(df_pareto_total,infile)
#        dfx = df_total[df_total.type == 'dram_weights no_d3']
#        df_total['utilization_norm'] = df_total['utilization'] / df_total['total_MAC_count']

        if parameter == 'utilization':
            plt.subplot(1,2,1)
        if parameter == 'spat_utilization':
            plt.subplot(2,2,2)
        if parameter == 'temp_utilization':
            plt.subplot(2,2,4)

        colors= plt.rcParams['axes.prop_cycle'].by_key()['color']
        lines = {0:'-',1:':',2:':',3:'-.'}
        legend_labels = []
        labels = []
        for ii_network, network in enumerate(df_pareto_total.network.unique()):
            for ii_c, c in enumerate(df_pareto_total.type.unique()):
                if c.split(' ')[0] == 'weights_on_chip':
                    d3 = c.split(' ')[1]
                    dfx_pareto = df_pareto_total[(df_pareto_total.network == network) & (df_pareto_total.type == c)]
                    dfx = df_total[(df_total.network == network) & (df_total.type == c)]
#                    plt.scatter(dfx['area'],dfx['utilization_norm'], c=colors[ii_network], ls=lines[ii_c],  alpha=0.05)
                    if c.split(" ")[1] == 'no_d3':
                        plt.plot(dfx_pareto['area'],dfx_pareto['utilization_norm'], c=colors[ii_network], ls=lines[ii_c], lw=2, alpha=0.7)
                    else:
                        plt.plot(dfx_pareto['area'],dfx_pareto['utilization_norm'], c=colors[ii_network], ls=lines[ii_c], lw=3)
                    if ii_c == 0:
                        legend_labels.append(Line2D([0],[0],color=colors[ii_network], ls='-'))
                        labels.append(f'{network}')
                    if ii_network == 3:
                        legend_labels.append(Line2D([0],[0],color='k', ls=lines[ii_c]))
                        labels.append(f'{c.split(" ")[1]}')




        plt.xscale('log')
        plt.ylabel(f'{parameter} [%]')
        if parameter != 'spat_utilization':
            plt.xlabel('Area [$\mu$m$^2$]')
        if parameter == 'utilization':
            plt.legend(legend_labels, labels,loc='lower left')
        plt.grid()
    plt.tight_layout()
    plt.show()
        


 
if __name__ == "__main__":
#    fig_plot()
    #fig_M_comparison()
    #fig_d3_comparison()
    fig_pareto_plot_m()
    #fig_pareto_plot_d1d2m()
    #fig_pareto_plot_d1d2d3m()
    #fig_pareto_plot_d1d2d3m_memory_shrink()
    #fig_plot_utilization_d1d2m()
    #fig_plot_utilization_d1d2d3m()
