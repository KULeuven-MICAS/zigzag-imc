import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import pandas as pd
import pickle
import os
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patheffects as pe
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shutil

import zigzag


def pareto_plot(parameter=None):
    area_net = {'mobilenet_v1':{'act':0.157,'weight':0.59},
            'resnet8':{'act':0.093, 'weight':0.213},
            'deepautoencoder':{'act':0.0058, 'weight':0.8},
            'ds_cnn':{'act':0.046, 'weight':0.066}}
            
    imc_type = 'dimc'
    for parameter in ['latency_total','latency_total','EDP']:
        df_total = pd.DataFrame()
        df_pareto_total = pd.DataFrame()
        fig,ax = plt.subplots(2,2, figsize=(15,10))
        for no_d3 in [True,False]:
            for network in ['ds_cnn','resnet8','deepautoencoder','mobilenet_v1']:
#            for network in ['mobilenet_v1']:
                if network in ['ds_cnn','resnet8','mobilenet_v1']:
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
#                df_onchip_pareto['D1_norm'] = df_onchip_pareto['D1'] / df_onchip_pareto['D1'].max()
#                df_onchip_pareto['D2_norm'] = df_onchip_pareto['D2'] / df_onchip_pareto['D2'].max()
#                df_onchip_pareto['D3_norm'] = df_onchip_pareto['D3'] / df_onchip_pareto['D3'].max()
#                df_onchip_pareto['M_norm'] = df_onchip_pareto['M'] / df_onchip_pareto['M'].max()
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
#                df_offchip_pareto['D1_norm'] = df_offchip_pareto['D1'] / df_offchip_pareto['D1'].max()
#                df_offchip_pareto['D2_norm'] = df_offchip_pareto['D2'] / df_offchip_pareto['D2'].max()
#                df_offchip_pareto['D3_norm'] = df_offchip_pareto['D3'] / df_offchip_pareto['D3'].max()
#                df_offchip_pareto['M_norm'] = df_offchip_pareto['M'] / df_offchip_pareto['M'].max()
                df_offchip_pareto = df_offchip_pareto.sort_values(by=['area',parameter],ascending=[True,True],ignore_index=True)

                df = pd.concat([df_onchip_weights, df_offchip_weights,df_all_weights])
                df_pareto = pd.concat([df_onchip_pareto, df_offchip_pareto, df_aw_pareto])
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

        breakpoint()   
        colors= plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig = make_subplots(2,2,subplot_titles=[x for x in df_total.network.unique()])
        df_pareto_total = df_pareto_total.sort_values(by=['area'],ascending=[True],ignore_index=True)
        for ii_network, network in enumerate(df_pareto_total.network.unique()):
            ax[(ii_network%2),ii_network//2].set_title(network)
            ax[(ii_network%2),ii_network//2].grid()
            ax[(ii_network%2),ii_network//2].set_xlabel('Area')
            ax[(ii_network%2),ii_network//2].set_ylabel(parameter)
            ax[(ii_network%2),ii_network//2].axvline(x=area_net[network]['act'])
            ax[(ii_network%2),ii_network//2].axvline(x=area_net[network]['weight']+area_net[network]['act'])
            for ii_c, type in enumerate(df_pareto_total.type.unique()):
                dfx =  df_total[(df_total.network == network) & (df_total.type == type)]
                dfx_pareto =  df_pareto_total[(df_pareto_total.network == network) & (df_pareto_total.type == type)]
                ax[(ii_network%2),ii_network//2].scatter(dfx['area'],dfx[parameter],label=type,c=colors[ii_c],alpha=0.2)
                ax[(ii_network%2),ii_network//2].plot(dfx_pareto['area'],dfx_pareto[parameter],c=colors[ii_c],lw=5)
                ax[(ii_network%2),ii_network//2].set_yscale('log')
                ax[(ii_network%2),ii_network//2].set_xscale('log')
            #    fig.add_trace(go.Scatter(x = dfx['area'],y=dfx[parameter],mode='markers',name=f'{network} {type}'), col=(ii_network%2)+1, row=(ii_network//2)+1)
            #    fig.add_trace(go.Scatter(x = dfx_pareto['area'],y=dfx_pareto[parameter],mode='markers+lines',name=f'{network} {type} pareto'), col=(ii_network%2)+1, row=(ii_network//2)+1)
            ax[(ii_network%2),ii_network//2].legend()
           # ax[(ii_network%2),ii_network//2].set_xlim(right=20)
        plt.tight_layout()
        plt.show()
        #breakpoint()
        #fig.update_xaxes(type='log')
        #fig.update_yaxes(type='log')
        #fig.show()

def network_area_req():
    df_total = pd.DataFrame()
    df_pareto_total = pd.DataFrame()
    area_df = []
    imc_type = 'aimc'
    for network in ['ds_cnn','resnet8','deepautoencoder','mobilenet_v1']:
        with open(f"outputs_{network}/{imc_type}/var/onchip_weights_{network}.pkl","rb") as infile:
            df_onchip_weights = pickle.load(infile)
        if network == 'resnet8':
            weight_sram_size = 77360
            act_sram_size = 32768 
        if network == 'deepautoencoder':
            weight_sram_size = 264192
            act_sram_size = 768 + 1024
        if network == 'ds_cnn':
            weight_sram_size = 22016
            act_sram_size = 16000
        if network == 'mobilenet_v1':
            weight_sram_size = 208112
            act_sram_size = 55296

        r = df_onchip_weights.iloc[0]
        area_df.append({'network':network, 'memory_size':act_sram_size, 'memory_area':r.act_sram_area,'memory_type':'Act'}) 
        area_df.append({'network':network, 'memory_size':weight_sram_size, 'memory_area':r.weight_sram_area,'memory_type':'Weight'}) 

    df = pd.DataFrame(area_df)
    width = 0.33
    x = np.arange(4)
    fig,ax = plt.subplots(2)
    for ii,ma in enumerate(['Memory size','Memory area']):
        mult = 0
        for att in ['Act','Weight']:
            dfx = df[df.memory_type == att]
            offset = width * mult
            if ii==0:
                rects = ax[ii].bar(x + offset, dfx['memory_size'],width, label=att)
            else:
                rects = ax[ii].bar(x + offset, dfx['memory_area'],width, label=att)

            ax[ii].bar_label(rects,fmt='%.1e',padding=3)
            mult +=1
        ax[ii].set_xticks(x+width, [x for x in df.network.unique()])
        ax[ii].set_title(ma)



    plt.show()
    breakpoint()



    fig = px.bar(df, 'network', 'memory_size', color='memory_type',barmode='group')
    fig.show()
    fig = px.bar(df, 'network', 'memory_area', color='memory_type',barmode='group')
    fig.show()

def pareto_points_energy_breakdown():
    imc_type = 'aimc'
    energy_bd = []
    for type in ['weights_on_chip d3']:# df.type.unique(): 
        for parameter in ['EDP','latency_total','energy_total']:
            for network in ['ds_cnn','resnet8','deepautoencoder','mobilenet_v1']:
                with open(f'outputs_{network}/{imc_type}/var/tinymlperf_pareto_{parameter}.pkl','rb') as infile:
                    df = pickle.load(infile)
                dfx = df[(df.network == network) & (df.type == type) & (df.parameter == parameter)]
                for cfg in dfx.cfg.unique():
                    dfxx = dfx[dfx.cfg == cfg]
                    ebb = dfxx.iloc[0]['energy_breakdown']
                    for key, energy in ebb.items():
                        energy_bd.append({'network':network,'type':type,'parameter':parameter,'cfg':cfg,'energy':energy,'energy_type':key})

    df_energy = pd.DataFrame(energy_bd)
    for type in df.type.unique(): 
        if type == 'weights_on_chip d3':
            for parameter in ['EDP']:# df.parameter.unique():
                for network in ['ds_cnn','resnet8','deepautoencoder','mobilenet_v1']:
                    dfx = df_energy[(df_energy.network == network) & (df_energy.type == type) & (df_energy.parameter == parameter)]
                    fig = px.bar(dfx,'cfg','energy',color='energy_type', title=f'{network} {parameter} {type}')
                    fig.show()

def pareto_plot_tops(parameter=None):
    area_net = {'mobilenet_v1':{'act':0.157,'weight':0.59},
            'resnet8':{'act':0.093, 'weight':0.213},
            'deepautoencoder':{'act':0.0058, 'weight':0.8},
            'ds_cnn':{'act':0.046, 'weight':0.066}}
    totalMACx = {'deepautoencoder':264192,'mobilenet_v1':7489664,'resnet8':12501632,'ds_cnn':2656768}
    imc_type = 'dimc'
          
    for parameter in ['TOPs', 'TOPs_mm2','TOPs_W']:
        df_total = pd.DataFrame()
        df_pareto_total = pd.DataFrame()
        fig,ax = plt.subplots(2,2, figsize=(15,10))
        for no_d3 in [True,False]:
            for network in ['ds_cnn','resnet8','deepautoencoder','mobilenet_v1']:
                if network in ['ds_cnn','resnet8']:
                    with open(f"outputs_{network}/{imc_type}/var/imc_weights_{network}.pkl","rb") as infile:
                        df_all_weights = pickle.load(infile)
                    df_all_weights['network'] = network

                    df_all_weights['EDP'] = df_all_weights['latency_total'] * df_all_weights['energy_total']
                    df_all_weights['totalMAC'] = df_all_weights['network'].apply(lambda x:totalMACx[x])
                    df_all_weights['TOPs_mm2'] = (df_all_weights['totalMAC'] / (df_all_weights['latency_total'] * 5e-9) / 1e12) / df_all_weights['area']
                    df_all_weights['TOPs'] = (df_all_weights['totalMAC'] / (df_all_weights['latency_total'] * 5e-9) / 1e12)
                    df_all_weights['TOPs_W'] =  (df_all_weights['totalMAC'])  / df_all_weights['energy_total']
 
                    df_all_weights['type'] = 'all_weights_on_chip'
                    df_aw_paretox = df_all_weights.sort_values(by=[parameter],ascending=[False],ignore_index=True)
                    df_aw_pareto = pd.DataFrame()
                    df_aw_pareto.append(df_aw_paretox.iloc[0])
                else:
                    df_aw_pareto = pd.DataFrame()
                    df_all_weights = pd.DataFrame()
                df_all_weights['network'] = network
                df_all_weights['type'] = 'all_weights_on_chip'
                df_aw_pareto['network'] = network
                df_aw_pareto['type'] = 'all_weights_on_chip'

                with open(f"outputs_{network}/{imc_type}/var/onchip_weights_{network}.pkl","rb") as infile:
                    df_onchip_weights = pickle.load(infile)
                df_onchip_weights['network'] = network
                df_onchip_weights['EDP'] = df_onchip_weights['latency_total'] * df_onchip_weights['energy_total']
                df_onchip_weights['type'] = 'weights_on_chip'
                df_onchip_weights['totalMAC'] = df_onchip_weights['network'].apply(lambda x:totalMACx[x])
                df_onchip_weights['TOPs_mm2'] = (df_onchip_weights['totalMAC'] / (df_onchip_weights['latency_total'] * 5e-9) / 1e12) / df_onchip_weights['area']
                df_onchip_weights['TOPs'] = (df_onchip_weights['totalMAC'] / (df_onchip_weights['latency_total'] * 5e-9))
                df_onchip_weights['TOPs_W'] =  (df_onchip_weights['totalMAC']) / df_onchip_weights['energy_total']
 
                df_onchip_pareto = df_onchip_weights.sort_values(by=[parameter],ascending=[False],ignore_index=True)
                df_onchip_pareto = df_onchip_pareto.groupby(by=['M'],sort=False, as_index=False).first()

                with open(f"outputs_{network}/{imc_type}/var/offchip_weights_{network}.pkl","rb") as infile:
                    df_offchip_weights = pickle.load(infile)
                df_offchip_weights['network'] = network
                df_offchip_weights['EDP'] = df_offchip_weights['latency_total'] * df_offchip_weights['energy_total']
                df_offchip_weights['totalMAC'] = df_offchip_weights['network'].apply(lambda x:totalMACx[x])
                df_offchip_weights['TOPs_mm2'] = (df_offchip_weights['totalMAC'] / (df_offchip_weights['latency_total'] * 5e-9) / 1e12) / df_offchip_weights['area']
                df_offchip_weights['TOPs'] = (df_offchip_weights['totalMAC'] / (df_offchip_weights['latency_total'] * 5e-9) / 1e12) 
                df_offchip_weights['TOPs_W'] = (df_offchip_weights['totalMAC']) / df_offchip_weights['energy_total'] 
                df_offchip_weights['type'] = 'dram_weights'
                df_offchip_pareto = df_offchip_weights.sort_values(by=[parameter],ascending=[False],ignore_index=True)
                df_offchip_pareto = df_offchip_pareto.groupby(by=['M'],sort=False, as_index=False).first()

                df = pd.concat([df_onchip_weights, df_offchip_weights,df_all_weights])
                df_pareto = pd.concat([df_onchip_pareto, df_offchip_pareto, df_aw_pareto])
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


        
        colors= plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig = make_subplots(2,2,subplot_titles=[x for x in df_total.network.unique()])
        df_pareto_total = df_pareto_total.sort_values(by=['area'],ascending=[True],ignore_index=True)
        for ii_network, network in enumerate(df_pareto_total.network.unique()):
            ax[(ii_network%2),ii_network//2].set_title(network)
            ax[(ii_network%2),ii_network//2].grid()
            ax[(ii_network%2),ii_network//2].set_xlabel('Area')
            ax[(ii_network%2),ii_network//2].set_ylabel(parameter)
            ax[(ii_network%2),ii_network//2].axvline(x=area_net[network]['act'])
            ax[(ii_network%2),ii_network//2].axvline(x=area_net[network]['weight']+area_net[network]['act'])
            for ii_c, type in enumerate(df_pareto_total.type.unique()):
                dfx =  df_total[(df_total.network == network) & (df_total.type == type)]
                dfx_pareto =  df_pareto_total[(df_pareto_total.network == network) & (df_pareto_total.type == type)]
                ax[(ii_network%2),ii_network//2].scatter(dfx['area'],dfx[parameter],label=type,c=colors[ii_c],alpha=0.5)
                ax[(ii_network%2),ii_network//2].scatter(dfx_pareto['area'],dfx_pareto[parameter],c=colors[ii_c],s=400,marker='*')#,edgecolors='k')
                for i,r in dfx_pareto.iterrows():
                    ax[(ii_network%2),ii_network//2].text(r['area'],r[parameter],str(int(r['M'])), color='black',path_effects=[pe.withStroke(linewidth=1, foreground='w')])#bbox={'facecolor':colors[ii_c],'pad':1,'edgecolor':None})
                ax[(ii_network%2),ii_network//2].set_yscale('log')
                ax[(ii_network%2),ii_network//2].set_xscale('log')
            #    fig.add_trace(go.Scatter(x = dfx['area'],y=dfx[parameter],mode='markers',name=f'{network} {type}'), col=(ii_network%2)+1, row=(ii_network//2)+1)
            #    fig.add_trace(go.Scatter(x = dfx_pareto['area'],y=dfx_pareto[parameter],mode='markers+lines',name=f'{network} {type} pareto'), col=(ii_network%2)+1, row=(ii_network//2)+1)
            ax[(ii_network%2),ii_network//2].legend()
            #ax[(ii_network%2),ii_network//2].set_xlim(right=10)
            #ax[(ii_network%2),ii_network//2].set_ylim(bottom=1e-3)
        plt.tight_layout()
        plt.show()
        #breakpoint()
        #fig.update_xaxes(type='log')
        #fig.update_yaxes(type='log')
        #fig.show()

def pareto_points_cfg():
    cfg_list = []
    imc_type = 'aimc'
    from tabulate import tabulate
    for parameter in ['energy_total']:
        for type in ['weights_on_chip d3']:#df.type.unique(): 
            for network in ['ds_cnn','resnet8','deepautoencoder','mobilenet_v1']:
                with open(f'outputs_{network}/{imc_type}/var/tinymlperf_pareto_{parameter}.pkl','rb') as infile:
                    df = pickle.load(infile)
                df['D1'] =df['D1'].astype(int)
                dfx = df[(df.network == network) & (df.type == type) & (df.parameter == parameter)]
                print(tabulate(dfx[[parameter,'area','M','D1','D2','D3','type','network']],headers='keys',tablefmt='psql'))

     
if __name__ == "__main__":
    pareto_plot()
#    network_area_req()
#    pareto_points_energy_breakdown()
    pareto_points_cfg()

