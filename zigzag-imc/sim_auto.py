import pickle
import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import main_dse_cp
import main_dse_aimc_cp
import peak_eval
import plot_figure
from termcolor import cprint
import math

def workload_eval_multi_processing(workloads, mp=True, clr=False):
    ''' 
    mp: enable multi-processing or not (True: enable; False: single processing)
    clr: whether remove existing .pkl files in the output folder (True: remove; False: not remove)
    '''
    import logging, time
    import itertools
    from multiprocessing import Process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s')    

    for imc_type in ['aimc_max','dimc']:
        START_ALL = time.time()
        for workload in workloads:
            cprint(f'imc_type: {imc_type}, workload: {workload}', color='white',on_color='on_green')
            if clr==True:
                os.system(f'rm -rf outputs/{imc_type}/{workload}/*.pkl') # clean output folder
            if mp==False: # clean input folder, only when not in multiprocessing mode, in case unintend file removal
                os.system(f'rm -rf inputs/tinyml/AIMC_*') # clean input folder
                os.system(f'rm -rf inputs/tinyml/DIMC_*') # clean input folder
                os.system(f'rm -rf inputs/tinyml/default_mapping*') # clean input folder
            os.makedirs(f'outputs/{imc_type}/{workload}/', exist_ok=True) # create input folder
            ## PRAMETER DEFINING SECTION
            area_target = [0.001] # no use anymore
            mem_ratio = [0.001] # no use anymore
            m_factor = [1]
            nb_cores = [1]
            act_pres = [8] # operand precision
            input_pres = [1 if imc_type=='dimc' else 2]
            di = [x for x in range(5, 11)]
            cols = di # nb_columns of SRAM cells
            weight_mem = ['dram'] # dram or sram for the top memory of weight
            cache_size = 256 # KB (only used for if condition below!)
            rc_list_initial = list(itertools.product(*[area_target, mem_ratio, m_factor, nb_cores, act_pres, input_pres, di, cols, weight_mem]))
            rc_list = []
            for e in rc_list_initial:
                # e[0]: area_target; e[1]: mem_ratio; e[2]: m; e[3]: nb_cores; e[4]: act_pres; e[5]: input_pes; e[6]: di; e[7]: cols; e[8]: weight_mem
                if e[-3] == e[-2] and e[6]*e[2]>=32 and e[7]>=8 and e[6]*e[2]*e[7]>=64*8 and cache_size*1024/(e[3]*e[6])>=32: # if di == cols
                    rc_list.append(e)
            ## MULTIPROCESSING SECTION
            if mp==True:
                chunks = min(100, len(rc_list)) # nb_processes
                rc_list_chunk = [rc_list[i:i + chunks] for i in range(0, len(rc_list), chunks)]
                for rcc in rc_list_chunk:
                    TIMEOUT = 36000
                    start = time.time()
                    if imc_type == 'dimc':
                        procs = [Process(target=main_dse_cp.mp_dp, args=(workload, imc_type, rc)) for rc in rcc]
                    else:
                        procs = [Process(target=main_dse_aimc_cp.mp_dp, args=(workload, imc_type, rc)) for rc in rcc]

                    for p in procs: p.start()
                    while time.time() - start <= TIMEOUT:
                        if not any(p.is_alive() for p in procs):
                            cprint(f'[MULTI-PROCESSING] FINISHED.',color='blue',on_color='on_green')
                            break
                        time.sleep(60)
                    else:
                        cprint('TIMED OUT - KILLING ALL PROCESSE', color='blue', on_color='on_red')
                        for p in procs:
                            p.terminate()
                            p.join()
                    for p in procs : p.join()
            else: ## SINGLE PROCESSING
                for rc in rc_list:
                    cprint(f'SINGLE PROCESSING, {workload}, {imc_type}, m: {rc[2]}, nb_macros: {rc[3]}, act_pres: {rc[4]}, input_pres: {rc[5]}, Di: {rc[6]}', color='white', on_color='on_green')
                    if imc_type=='dimc':
                        main_dse_cp.mp_dp(workload,imc_type,rc)
                    else:
                        main_dse_aimc_cp.mp_dp(workload,imc_type,rc)
                cprint(f'[SINGLE PROCESSING] FINISHED.',color='white',on_color='on_green')
            TIME_TOTAL = round( (time.time() - START_ALL)/60, 1)
            cprint(f'[TIME RECORD] TOTAL TIME (MINUTES): {TIME_TOTAL}. WORKLOADS: {workloads} ', color='white',on_color='on_green')


def workload_output_read(workloads):
    data_vals = []
    ops_workloads = {'ae': 532512, 'ds-cnn': 5609536, 'mobilenet': 15907840, 'resnet8': 25302272}
    
    for ii_a, a in enumerate(['aimc_max', 'dimc']):
        for ii_c, workload in enumerate(workloads):
            directory = f'outputs/{a}/{workload}/'
            # read imc data
            for filename in os.listdir(directory):
                f = os.path.join(directory, filename)
                if os.path.isfile(f):
                    if f.split('.')[-1] == 'pkl':
                        with open(f,'rb') as infile:
                            data = pickle.load(infile)
                        if len(data[1]) == 0: # no output from zigzag (due to 0.5 util restriction. This issue has been fixed.)
                            print(f'[WARNING] no output detected for some cases')
                            continue
                        dx = data[0]
                        cme_list = [v for k,v in data[1].items()]
                        en_total = sum([v.energy_total for v in cme_list])
                        lat_total = sum([v.latency_total0 for v in cme_list])
                        lat_ideal_total = sum([v.ideal_temporal_cycle for v in cme_list])
                        lat_stall_total = lat_total - lat_ideal_total # about = weight_loading_cycles
                        dx['workload'] = workload
                        dx['ops'] = ops_workloads[workload]
                        dx['m'] = str(dx['cells_per_mult'])
                        dx['cores'] = str(dx['macros'])
                        # dx['rows']: nb_rows per core (already exist)
                        # dx['cols']: nb_cols per core (already exist)
                        dx['energy_total'] = en_total # pJ
                        dx['latency_total'] = lat_total # unit: cycle
                        dx['latency_ideal_total'] = lat_ideal_total # unit: cycle
                        dx['latency_stall_total'] = lat_stall_total # unit: cycle
                        dx['delay_total'] = data[0]['tclk'] * lat_total # unit: ns
                        dx['topsw_workload'] = dx['ops']/dx['energy_total']
                        dx['tops_workload'] = dx['ops']/dx['delay_total']/1000
                        dx['area_imc'] = data[0]['area'] * data[0]['macros'] #mm2
                        dx['area_cache'] = data[0]['l2_sram_area'] # mm2
                        dx['area_weight_mem'] = 0 # mm2
                        dx['area_total'] = dx['area_imc'] + dx['area_cache'] + dx['area_weight_mem'] #mm2
                        dx['topsmm2_workload'] = dx['tops_workload']/dx['area_total']
                        dx['cme'] = cme_list
                        dx['spatial_mapping'] = [f'{cme_list[x].spatial_mapping}' for x in range(0, len(cme_list))]
                        dx['temporal_mapping'] = [f'{cme_list[x].temporal_mapping}' for x in range(0, len(cme_list))]
                        dx['energy_mem'] = [v.energy_total for v in cme_list]
                        dx['energy_mem_breakdown'] = [v.energy_breakdown for v in cme_list]
                        dx['energy_mem_breakdown_further'] = [v.energy_breakdown_further for v in cme_list]
                        dx['energy_mac'] = [f'{cme_list[x].MAC_energy}' for x in range(0, len(cme_list))]
                        dx['energy_mac_breakdown'] = [f'{cme_list[x].MAC_energy_breakdown}' for x in range(0, len(cme_list))]
                        en_mem = 0
                        en_mem_weight_update = 0
                        en_mem_weight_dram = 0
                        en_mem_in_cache = 0
                        en_mem_out_cache = 0
                        en_mem_in_reg = 0
                        en_mem_out_reg = 0
                        for x in cme_list: # x: layer_id
                            k = x.energy_breakdown.keys() # k: O,B,A or O,W,I
                            for e in k:
                                en_mem += sum(x.energy_breakdown[e])
                            en_mem_out_reg += x.energy_breakdown['O'][0]
                            try:
                                en_mem_out_cache += x.energy_breakdown['O'][1]
                            except:
                                print('cache level is removed for O')
                            if 'A' in k:
                                en_mem_in_reg += x.energy_breakdown['A'][0]
                                try:
                                    en_mem_in_cache += x.energy_breakdown['A'][1]
                                except:
                                    print('cache level is removed for A')
                            elif 'I' in k:
                                en_mem_in_reg += x.energy_breakdown['I'][0]
                                try:
                                    en_mem_in_cache += x.energy_breakdown['I'][1]
                                except:
                                    print('cache level is removed for I')
                            else:
                                breakpoint() # abnormal

                            if 'B' in k:
                                en_mem_weight_update += x.energy_breakdown['B'][0]
                                try:
                                    en_mem_weight_dram += x.energy_breakdown['B'][1]
                                except: # dram level is removed
                                    en_mem_weight_dram += 0
                            elif 'W' in k:
                                en_mem_weight_update += x.energy_breakdown['W'][0]
                                try:
                                    en_mem_weight_dram += x.energy_breakdown['W'][1]
                                except: # dram level is removed
                                    en_mem_weight_dram += 0
                            else:
                                breakpoint() # abnormal
                        dx['energy_mem_breakdown_weight_update'] = en_mem_weight_update
                        dx['energy_mem_breakdown_weight_dram'] = en_mem_weight_dram
                        dx['energy_mem_breakdown_out_cache'] = en_mem_out_cache
                        dx['energy_mem_breakdown_out_reg'] = en_mem_out_reg
                        dx['energy_mem_breakdown_in_cache'] = en_mem_in_cache
                        dx['energy_mem_breakdown_in_reg'] = en_mem_in_reg
                        dx['energy_mac_total'] = sum([v.MAC_energy for v in cme_list])
                        dx['energy_mem_total'] = en_mem
                        data_vals.append(dx)
    
    
    df = pd.DataFrame(data_vals)
    #fig = px.scatter(df[df.m=='1'],x='cores',y='topsw_workload',symbol='accelerator_type',color='cores')
    #fig = px.scatter(df[df.m=='1'],x='topsmm2_workload',y='topsw_workload',symbol='accelerator_type',color='cores',hover_data=['rows','cols','r_in_chl','r_out_chl','cores','m','tclk'],symbol_map={'aimc_max':'circle-open','dimc':'circle'})
    #breakpoint()
    return df

def workload_vs_system(simulation='off', en_layer='off', en_workload='off', key='imc'):
    fontsize = 10
    ''' plot peak/workload TOP/s/W and TOP/s/mm2 vs. IMC size for AIMC/DIMC '''
    '''
    :simulation: if do the simulation for peak workload (required if it has been done before or in other files) (default: 'off')
    :en_layer: switch to show layer performance
    :en_workload: switch to show workload performance (default: 'off')
    '''
    ''' simulation for peak '''
    if simulation == 'on':
        if key == 'imc':
            rows_list = [2**x for x in range(5,11)]
            cores_list = [1]
            m_list = [1]
            activation_precision_list = [8]
            input_precision_list = [1,2] # 1 for DIMC, 2 for AIMC
        elif key == 'cores':
            rows_list = [32,64,128,256]
            cores_list = [1,4,16,64]
            m_list = [1]
            activation_precision_list = [8]
            input_precision_list = [1,2] # 1 for DIMC, 2 for AIMC
        else:
            rows_list = [256, 512, 1024, 2048]
            cores_list = [1]
            m_list = [1,4,16,64]
            activation_precision_list = [8]
            input_precision_list = [1,2] # 1 for DIMC, 2 for AIMC
        imc_list = ['aimc_max', 'dimc'] # 'aimc_max': aimc with dynamic Res_adc; 'aimc_min': aimc with fixed Res_adc=activation precision; 'dimc': dimc
        peak_eval.peak_eval(rows_list, cores_list, m_list, activation_precision_list, input_precision_list, imc_list)

    ''' dfr: workload data '''
    #colors = ['orange','mediumseagreen', u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    colors = ['orange', 'mediumseagreen', u'#27E1C1', u'#ff7f0e', u'#CCD5AE', 'plum', u'#BDE0FE', u'#CDB4DB', u'#FAD2E1']
    if en_workload != 'off': # color for geo mean
        colors[6] = 'gold'

    df =  pd.read_pickle('peak_eval_256kB.pickle') # LOAD PEAK DATA (need update peak_eval.py with different imc size)
    df['imc'] = df.rows.astype(str)+f"$\\times$"+df.cols.astype(str)
    df = df[(df.act_precision== '8') & ((df.input_precision == '2')&(df.imc_type=='aimc_max') | ((df.input_precision=='1')&(df.imc_type=='dimc'))) & (df.rows==df.cols)]

    if key == 'imc':
        df = df[(df.cores == '1') & (df.m == '1')]
    elif key == 'cores':
        df = df[(df.m == '1')]
    else:
        df = df[df.cores == '1']

    fig, axs = plt.subplots(nrows=1, ncols=2)
    #axs[1].set_xlabel(f'IMC size (#rows$\\times$#columns)',color='black')
    #fig.text(0.5,0,'IMC size (#rows$\\times$#columns)',ha='center')
    if key=='imc':
        fig.text(0.5,0,'IMC size (#rows: D$_i$, #columns: D$_o\\cdot$B$_w$)', fontsize=fontsize,ha='center')
    elif key=='cores':
        fig.text(0.5,0,'#cores', fontsize=fontsize,ha='center')
    else:
        fig.text(0.5,0,'M', fontsize=fontsize,ha='center')
    axs[0].set_ylabel(f'TOP/s/W',color='black', fontsize=fontsize)
    axs[0].set_yscale('log')    
    axs[0].set_axisbelow(True)

    axs[1].set_axisbelow(True)
    axs[0].grid(which='both')
    axs[1].grid(which='both')


    #plt.title(f'TOP/s/W (@{dfr.iloc[0].workload})')
    imc_marker = ['o','^']

    ''' plot peak '''
    def plot_peak(y='topsw', axs_idx=0,ltype='system', lb_system='on', p_m_size=7, key='imc'):
        '''
        : p_m_size: peak performance marker size when comparing with wokrload performance
        '''
        for ii_a, a in enumerate(['aimc_max','dimc']):
            if a == 'aimc_max':
                b = 'AIMC'
            else:
                b = 'DIMC'
            if lb_system == 'on':
                label_system = ' system'
            else:
                label_system = ''
            if key == 'imc':
                dfx = df[(df.imc_type==a)]
                imc = dfx.imc.to_numpy()
            elif key == 'cores':
                dfx = df[(df.imc_type==a)].sort_values(by='rows', ascending=False)
                imc = dfx.cores.to_numpy() # x axis
            else:
                dfx = df[(df.imc_type==a)]
                imc = dfx.m.to_numpy() # x axis
            #dfx = df[(df.imc_type==a)]
            rows = dfx.rows.to_numpy()
            topsw_macro = dfx.topsw_macro.to_numpy()
            topsw_system = dfx.topsw_system.to_numpy()
            topsmm2_macro = dfx.topsmm2_macro.to_numpy()
            topsmm2_system = dfx.topsmm2_system.to_numpy()
            if key == 'imc':
                imc = dfx.imc.to_numpy()
            elif key == 'cores':
                imc = dfx.cores.to_numpy() # x axis
            else:
                imc = dfx.m.to_numpy() # x axis

            if y=='topsw':
                y_axis_system = topsw_system
                y_axis_macro = topsw_macro
            else:
                y_axis_system = topsmm2_system
                y_axis_macro = topsmm2_macro
            if axs_idx==0:
                if ltype=='system':
                    if lb_system == 'on':
                        axs[0].plot(imc, y_axis_system, marker=imc_marker[ii_a],color=colors[1],linestyle='-.',label=f'{b} (peak{label_system})', markeredgewidth=1.0, markeredgecolor='black')
                    else: # remove its label and color when comparing with workload, change marker size
                        axs[0].plot(imc, y_axis_system, marker=imc_marker[ii_a],color=colors[1],linestyle='-',label=f'{b} (peak{label_system})', markeredgewidth=1.0, markeredgecolor='black', markersize=p_m_size)
                else:
                    axs[0].plot(imc, y_axis_macro, marker=imc_marker[ii_a], color=colors[0],linestyle=':', label=f'{b} (peak macro)', markeredgewidth=1.0, markeredgecolor='black')
            else:
                if ltype=='system':
                    if lb_system == 'on':
                        axs[1].plot(imc, y_axis_system, marker=imc_marker[ii_a],color=colors[1],linestyle='-.',label=f'{b} (peak{label_system})', markeredgewidth=1.0, markeredgecolor='black')
                    else:
                        axs[1].plot(imc, y_axis_system, marker=imc_marker[ii_a],color=colors[1],linestyle='-',label=f'{b} (peak{label_system})', markeredgewidth=1.0, markeredgecolor='black', markersize=p_m_size)
                else:
                    axs[1].plot(imc, y_axis_macro, marker=imc_marker[ii_a], color=colors[0],linestyle=':', label=f'{b} (peak macro)', markeredgewidth=1.0, markeredgecolor='black')
        
    ''' plot workload '''
    def plot_layer(workload=['ae'], layer_idx=0, nb_ops=163840, y='topsw', axs_idx = 0, color_idx=2, ltype='fc', i_dfr=None, alpha=1, key='imc'):
        if type(i_dfr) == type(None): # will fetch the data myself
            dfr = plot_figure.layer_info_read(workload, layer_idx, nb_ops) # fetch data
        else: # data is provided
            dfr = i_dfr 
        dfr['imc'] = dfr.rows.astype(str)+f"$\\times$"+dfr.cols.astype(str)
        dfr = dfr[(dfr.act_pres==8) & ( ((dfr.input_precision==2)&(dfr.accelerator_type=='aimc_max')) | ((dfr.input_precision==1)&(dfr.accelerator_type=='dimc')) )].sort_values(by='rows')
        if key == 'imc':
            dfr = dfr[(dfr.cores=='1') & (dfr.m=='1')]
        elif key == 'cores':
            dfr = dfr[(dfr.m=='1')].sort_values(by='rows',ascending=False)
        else:
            dfr = dfr[(dfr.cores=='1')]
        for ii_a, a in enumerate(['aimc_max','dimc']):
            if a == 'aimc_max':
                b = 'AIMC'
            else:
                b = 'DIMC'
            dfx = dfr[dfr.accelerator_type==a]
            topsw_workload = dfx.topsw_workload.to_numpy()
            topsmm2_workload = dfx.topsmm2_workload.to_numpy()
            if key == 'imc':
                imc = dfx.imc.to_numpy()
            elif key == 'cores':
                imc = dfx.cores.to_numpy()
            else:
                imc = dfx.m.to_numpy()

            if y=='topsw':
                y_axis = topsw_workload
            elif y=='topsmm2':
                y_axis = topsmm2_workload
            elif y=='topsw_geo_mean':
                y_axis = dfx.topsw_geo_mean.to_numpy()
            elif y=='topsmm2_geo_mean':
                y_axis = dfx.topsmm2_geo_mean.to_numpy()
            else:
                breakpoint() # should never triggered
            if axs_idx == 0:
                if y!='topsw_geo_mean':
                    if type(i_dfr) == type(None): # true when plotting for single layer
                        axs[0].plot(imc, y_axis, marker=imc_marker[ii_a],color=colors[color_idx],linestyle=':',label=f'{b} ({ltype})', markeredgewidth=1.0, markeredgecolor='black', alpha=alpha)
                    else:
                        axs[0].plot(imc, y_axis, marker=imc_marker[ii_a],color=colors[color_idx],linestyle=':',label=f'{b} ({ltype})', markeredgewidth=1.0, markeredgecolor='black', alpha=alpha, markersize=5)
                else: # increase marker size
                    axs[0].plot(imc, y_axis, marker=imc_marker[ii_a],markersize=8,color='black',linestyle='-',label=f'{b} ({ltype})', markeredgewidth=1.0, markeredgecolor='black', alpha=alpha, markerfacecolor=colors[color_idx])
            else:
                if y!='topsmm2_geo_mean':
                    if type(i_dfr) == type(None): # true when plotting for single layer
                        axs[1].plot(imc, y_axis, marker=imc_marker[ii_a],color=colors[color_idx],linestyle=':',label=f'{b} ({ltype})', markeredgewidth=1.0, markeredgecolor='black',alpha=alpha)
                    else:
                        axs[1].plot(imc, y_axis, marker=imc_marker[ii_a],color=colors[color_idx],linestyle=':',label=f'{b} ({ltype})', markeredgewidth=1.0, markeredgecolor='black',alpha=alpha, markersize=5)
                else:
                    axs[1].plot(imc, y_axis, marker=imc_marker[ii_a], markersize=8, color='black',linestyle='-',label=f'{b} ({ltype})', markeredgewidth=1.0, markeredgecolor='black',alpha=alpha, markerfacecolor=colors[color_idx])

    # plot peak topsw
    if en_layer == 'off' and en_workload == 'off':
        plot_peak(y='topsw', axs_idx=0, ltype='system', p_m_size=6, key=key)
        plot_peak(y='topsw', axs_idx=0, ltype='macro', p_m_size=6, key=key)
    else:
        if en_workload != 'off': # increase the marker size of peak
            plot_peak(y='topsw', axs_idx=0, ltype='system', lb_system='off', p_m_size=6, key=key)
        else:
            plot_peak(y='topsw', axs_idx=0, ltype='system', lb_system='off', p_m_size=6, key=key)
    if en_layer != 'off':
        # plot workload topsw
        # dense layer
        workload, layer_idx, nb_ops = ['ae'], 0, 163840 # C=640, K=128
        plot_layer(workload, layer_idx, nb_ops, y='topsw', axs_idx = 0, color_idx=2, ltype='fc', key=key)
        # pw layer
        #workload, layer_idx, nb_ops = ['ds-cnn'], 2, 1024000 # C=K=64, OX/OY=25/5
        workload, layer_idx, nb_ops = ['mobilenet'], 10, 1179648 # C=K=64, OX=OY=12
        plot_layer(workload, layer_idx, nb_ops, y='topsw', axs_idx = 0, color_idx=3, ltype='pw', key=key)
        # dw layer
        workload, layer_idx, nb_ops = ['ds-cnn'], 1, 144000 # G=64, FX=FY=3, OX/OY=25/5
        plot_layer(workload, layer_idx, nb_ops, y='topsw', axs_idx = 0, color_idx=4, ltype='dw', key=key)
        # conv layer
        #workload, layer_idx, nb_ops = ['ds-cnn'], 0, 640000 # K/C=64/1, FX/FY=10/4, OX/OY=25/5
        workload, layer_idx, nb_ops = ['resnet8'], 2, 4718592 # K/C=16/16, FX/FY=3/3, OX/OY=32/32
        plot_layer(workload, layer_idx, nb_ops, y='topsw', axs_idx = 0, color_idx=5, ltype='conv', key=key)

    if en_workload != 'off':
        workloads = [['ds-cnn'],['mobilenet'],['resnet8'],['ae']]
        nm_workloads = ['DS-CNN','MobileNet_v1','ResNet8','AutoEncoder']
        nets = []
        alpha = 1
        for ii_wk, workload in enumerate(workloads):
            net = workload_output_read(workload).sort_values(by='rows')
            plot_layer(i_dfr=net, y='topsw', axs_idx=0, color_idx=2+ii_wk, ltype=f'{nm_workloads[ii_wk]}', alpha=alpha, key=key)
            nets.append(net)
        # geometry mean
        geo_df = nets[0]
        topsw_product = np.prod([x.topsw_workload.to_numpy() for x in nets], axis=0)
        topsmm2_product = np.prod([x.topsmm2_workload.to_numpy() for x in nets], axis=0)
        topsw_geo_mean = np.power(topsw_product, 1/len(nets))
        topsmm2_geo_mean = np.power(topsmm2_product, 1/len(nets))
        geo_df['topsw_geo_mean'] = topsw_geo_mean 
        geo_df['topsmm2_geo_mean'] = topsmm2_geo_mean
        plot_layer(workload=None, layer_idx=None, nb_ops=None, i_dfr=geo_df, y='topsw_geo_mean', axs_idx=0, color_idx=6, ltype='mean', key=key)

    # put legend here, to aviod repetive legend for the 2nd subplot
    if en_workload != 'off':
        fig.legend(loc='upper right', ncol=7, frameon=False)
    elif en_layer != 'off':
        fig.legend(loc='upper right', ncol=5, frameon=False)
    else: # only peak
        fig.legend(loc='upper right', ncol=4, frameon=False)


    if en_workload != 'off':
        workloads = [['ds-cnn'],['mobilenet'],['resnet8'],['ae']]
        nm_workloads = ['DS-CNN','MobileNet_v1','ResNet8','DeepAutoEncoder']
        for ii_wk, workload in enumerate(workloads):
            plot_layer(i_dfr=nets[ii_wk], y='topsmm2', axs_idx=1, color_idx=2+ii_wk, ltype=f'{nm_workloads[ii_wk]}', alpha=alpha, key=key)
        plot_layer(workload=None, layer_idx=None, nb_ops=None, i_dfr=geo_df, y='topsmm2_geo_mean', axs_idx=1, color_idx=6, ltype='mean', key=key)

    # topsmm2
    axs[1].set_ylabel(f'TOP/s/mm$^2$',color='black', fontsize=fontsize)
    axs[1].set_yscale('log')
    # plot peak topsmm2
    if en_layer == 'off' and en_workload == 'off':
        plot_peak(y='topsmm2', axs_idx=1, ltype='system', p_m_size=6, key=key)
        plot_peak(y='topsmm2', axs_idx=1, ltype='macro', p_m_size=6, key=key)
    else:
        if en_workload != 'off':
            plot_peak(y='topsmm2', axs_idx=1, ltype='system', lb_system='off', p_m_size=6, key=key)
        else:
            plot_peak(y='topsmm2', axs_idx=1, ltype='system', lb_system='off', p_m_size=6, key=key)

    if en_layer != 'off':
        # plot workload
        # dense layer
        workload, layer_idx, nb_ops = ['ae'], 0, 163840 # C=640, K=128
        plot_layer(workload, layer_idx, nb_ops, y='topsmm2', axs_idx=1, color_idx=2, ltype='fc layer', key=key)
        # pw layer
        #workload, layer_idx, nb_ops = ['ds-cnn'], 2, 1024000 # C=K=64, OX/OY=25/5
        workload, layer_idx, nb_ops = ['mobilenet'], 10, 1179648 # C=K=64, OX=OY=12
        plot_layer(workload, layer_idx, nb_ops, y='topsmm2', axs_idx=1, color_idx=3, ltype='pw layer', key=key)
        # dw layer
        workload, layer_idx, nb_ops = ['ds-cnn'], 1, 144000 # G=64, FX=FY=3, OX/OY=25/5
        plot_layer(workload, layer_idx, nb_ops, y='topsmm2', axs_idx = 1, color_idx=4, ltype='dw layer', key=key)
        # conv layer
        #workload, layer_idx, nb_ops = ['ds-cnn'], 0, 640000 # K/C=64/1, FX/FY=10/4, OX/OY=25/5
        workload, layer_idx, nb_ops = ['resnet8'], 2, 4718592 # K/C=16/16, FX/FY=3/3, OX/OY=32/32
        plot_layer(workload, layer_idx, nb_ops, y='topsmm2', axs_idx = 1, color_idx=5, ltype='conv layer', key=key)
        
    if key=='imc':
        r_angle = 25
    else:
        r_angle = 0
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=r_angle, fontsize=fontsize)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=r_angle, fontsize=fontsize)
    plt.tight_layout()
    plt.show()

def plt_multiple_latency_breakdown(key='imc'):
    def plot_latency_breakdown(workload, layer_idx, nb_ops, ax1, ltype, key='imc'):
        ''' plot latency breakdown of a single layer (unit: cycle) '''
        ''' dfr: workload data '''
        dfr = plot_figure.layer_info_read(workload, layer_idx, nb_ops)
        dfr['imc'] = dfr.rows.astype(str)+f"$\\times$"+dfr.cols.astype(str)
        dfr = dfr[(dfr.act_pres==8) & ( ((dfr.input_precision==2)&(dfr.accelerator_type=='aimc_max')) | ((dfr.input_precision==1)&(dfr.accelerator_type=='dimc')) )].sort_values(by='rows')
        if key == 'imc':
            dfr = dfr[(dfr.cores=='1')&(dfr.m=='1') & (dfr.rows == dfr.cols)]
        elif key == 'cores':
            dfr = dfr[(dfr.m=='1') & (dfr.rows == dfr.cols)]
        else:
            dfr = dfr[(dfr.cores=='1')]
        #colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#BDE0FE', u'#7f7f7f', u'#bcbd22', u'#17becf']
        colors = [u'#633E7A', u'#ff7f0e', u'#CCD5AE', u'#FEFAE0', u'#27E1C1', u'#BDE0FE', u'#CDB4DB', u'#FAD2E1']
        #colors_twin = ['violet','mediumseagreen']
        colors_twin = ['red','blue','green']

        width = 0.35
        line_ax1 = [] # for ease of legend generation
        for ii_a, a in enumerate(['aimc_max','dimc']):
            df = dfr[dfr.accelerator_type==a]
            if key == 'imc':
                df = df.sort_values(by='rows')
                labels = df.imc.to_numpy() # x label
                #ax1.set_xlabel(f'IMC size')
            elif key == 'cores':
                df = df.sort_values(by='rows', ascending=False)
                labels = df.cores.to_numpy() # x label
                #ax1.set_xlabel(f'cores')
            elif key == 'm':
                df = df.sort_values(by='rows')
                labels = df.m.to_numpy() # x label
                #ax1.set_xlabel(f'M')
            else:
                breakpoint() # key incorrect, debug required

            # Create positions for the bars on the x-axis
            x_pos = np.arange(len(labels))
            # Plot the bars
            base = [0 for x in range(0, len(labels))]
            values = df.latency_stall_total.to_numpy() # cycles
            if ii_a == 0:
                cctype = f'CC$_{{stall}}$'
            else:
                cctype = None # avoid creating duplicate label in legends

            if ii_a == 0:
                ii_pos = -1
            else:
                ii_pos = 1
            bar = ax1.bar(x_pos+width/2*ii_pos, values, width, label=cctype, bottom = base, color = colors[0], edgecolor='black')
            if ii_a == 0:
                line_ax1.append(bar)
            base += values

            values = df.latency_ideal_total.to_numpy() # cycles
            if ii_a == 0:
                cctype = f'CC$_{{mac}}$'
            else:
                cctype = None # avoid creating duplicate label in legends

            if ii_a == 0:
                ii_pos = -1
            else:
                ii_pos = 1
            bar = ax1.bar(x_pos+width/2*ii_pos, values, width, label=cctype, bottom = base, color = colors[1], edgecolor='black')
            if ii_a == 0:
                line_ax1.append(bar)

        ax1.set_xticks(x_pos)
        if key == 'imc':
            r_angle = 45
        else:
            r_angle = 0
        ax1.set_xticklabels(labels, rotation=r_angle)
        ax1.set_title(f'{ltype} layer')
        return line_ax1

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    if key == 'imc':
        fig.text(0.5,0,f'IMC size (#rows: D$_i$, #columns: D$_o\\cdot$B$_w$)',ha='center') # x label
    elif key == 'cores':
        fig.text(0.5,0,f'#cores',ha='center') # x label
    else: # key == 'm'
        fig.text(0.5,0,f'M',ha='center') # x label
    fig.text(0.02,0.5,'Latency breakdown [cycles]',va='center', rotation='vertical') # y label
    # dense layer
    workload, layer_idx, nb_ops = ['ae'], 0, 163840 # C=640, K=128
    lines = plot_latency_breakdown(workload, layer_idx, nb_ops, axs[0][0], ltype='fc', key=key)
    # pw layer
    #workload, layer_idx, nb_ops = ['ds-cnn'], 2, 1024000 # C=K=64, OX/OY=25/5
    workload, layer_idx, nb_ops = ['mobilenet'], 10, 1179648 # C=K=64, OX=OY=12
    plot_latency_breakdown(workload, layer_idx, nb_ops, axs[0][1], ltype='pw', key=key)
    # dw layer
    workload, layer_idx, nb_ops = ['ds-cnn'], 1, 144000 # G=64, FX=FY=3, OX/OY=25/5
    plot_latency_breakdown(workload, layer_idx, nb_ops, axs[1][0], ltype='dw', key=key)
    # conv layer
    #workload, layer_idx, nb_ops = ['ds-cnn'], 0, 640000 # K/C=64/1, FX/FY=10/4, OX/OY=25/5
    workload, layer_idx, nb_ops = ['resnet8'], 1, 4718592 # K/C=16/16, FX/FY=3/3, OX/OY=32/32
    plot_latency_breakdown(workload, layer_idx, nb_ops, axs[1][1], ltype='conv', key=key)
    # Set the labels and title
    #plt.title(f'Latency (cycles) breakdown (different {key} @ {dfr.iloc[0].workload})')
    # Add legend
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc='upper center', ncol=6, frameon=False)
    # Show the chart
    plt.tight_layout()
    plt.show()

def plt_multiple_energy_breakdown(key='imc', ax2_enable = 'on'):
    # key: 'imc', 'cores', 'm'
    # ax2_enable: show macro utilization or not ('on': show; else: not show)

    def plot_e_breakdown(workload, layer_idx, nb_ops, ax1, ltype, key='imc', ax2_enable='on'):
        ''' dfr: workload data '''
        dfr = plot_figure.layer_info_read(workload, layer_idx, nb_ops)
        dfr['imc'] = dfr.rows.astype(str)+f"$\\times$"+dfr.cols.astype(str)
        dfr = dfr[(dfr.act_pres==8) & ( ((dfr.input_precision==2)&(dfr.accelerator_type=='aimc_max')) | ((dfr.input_precision==1)&(dfr.accelerator_type=='dimc')) )].sort_values(by='rows')
        if key == 'imc':
            dfr = dfr[(dfr.cores=='1')&(dfr.m=='1') & (dfr.rows == dfr.cols)]
        elif key == 'cores':
            dfr = dfr[(dfr.m=='1') & (dfr.rows == dfr.cols)]
        else:
            dfr = dfr[(dfr.cores=='1')]
        #colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#BDE0FE', u'#7f7f7f', u'#bcbd22', u'#17becf']
        colors = [u'#633E7A', u'#ff7f0e', u'#CCD5AE', u'#FEFAE0', u'#27E1C1', u'#BDE0FE', u'#CDB4DB', u'#FAD2E1']
        #colors_twin = ['violet','mediumseagreen']
        colors_twin = ['red','blue','green']

        width = 0.35
        line_ax1 = [] # for ease of legend generation
        for ii_a, a in enumerate(['aimc_max','dimc']):
            df = dfr[dfr.accelerator_type==a]
            if key == 'imc':
                df = df.sort_values(by='rows')
                labels = df.imc.to_numpy() # x label
                #ax1.set_xlabel(f'IMC size')
            elif key == 'cores':
                df = df.sort_values(by='rows', ascending=False)
                labels = df.cores.to_numpy() # x label
                #ax1.set_xlabel(f'cores')
            elif key == 'm':
                df = df.sort_values(by='rows')
                labels = df.m.to_numpy() # x label
                #ax1.set_xlabel(f'M')
            else:
                breakpoint() # key incorrect, debug required

            # Create positions for the bars on the x-axis
            x_pos = np.arange(len(labels))
            # Plot the bars
            base = [0 for x in range(0, len(labels))]
            values = df.energy_mem_breakdown_weight_update.to_numpy()/(nb_ops/2) # pJ/MAC
            if ii_a == 0:
                etype = f'E$_{{weight\_update}}$'
            else:
                etype = None # avoid creating duplicate label in legends

            if ii_a == 0:
                ii_pos = -1
            else:
                ii_pos = 1
            bar = ax1.bar(x_pos+width/2*ii_pos, values, width, label=etype, bottom = base, color = colors[0], edgecolor='black')
            if ii_a == 0:
                line_ax1.append(bar)
            base += values
            values = df.energy_mem_breakdown_weight_dram.to_numpy()/(nb_ops/2) # pJ/MAC
            if ii_a == 0:
                etype = f'E$_{{weight\_dram\_access}}$'
            else:
                etype = None
            bar = ax1.bar(x_pos+width/2*ii_pos, values, width, label=etype, bottom = base, color = colors[1], edgecolor='black')
            if ii_a == 0:
                line_ax1.append(bar)
            base += values
            values = df.energy_mem_breakdown_out_cache.to_numpy()/(nb_ops/2) # pJ/MAC
            if ii_a == 0:
                etype = f'E$_{{out\_regs\_to\_cache}}$'
            else:
                etype = None
            bar = ax1.bar(x_pos+width/2*ii_pos, values, width, label=etype, bottom = base, color = colors[2], edgecolor='black')
            if ii_a == 0:
                line_ax1.append(bar)
            base += values

            values = df.energy_mem_breakdown_out_reg.to_numpy()/(nb_ops/2) # pJ/MAC
            if ii_a == 0:
                etype = f'E$_{{out\_regs\_update}}$'
            else:
                etype = None
            bar = ax1.bar(x_pos+width/2*ii_pos, values, width, label=etype, bottom = base, color = colors[3], edgecolor='black')
            if ii_a == 0:
                line_ax1.append(bar)
            base += values
            
            values = df.energy_mem_breakdown_in_cache.to_numpy()/(nb_ops/2) # pJ/MAC
            if ii_a == 0:
                etype = f'E$_{{cache\_to\_in\_regs}}$'
            else:
                etype = None
            bar = ax1.bar(x_pos+width/2*ii_pos, values, width, label=etype, bottom = base, color = colors[4], edgecolor='black')
            if ii_a == 0:
                line_ax1.append(bar)
            base += values

            values = df.energy_mem_breakdown_in_reg.to_numpy()/(nb_ops/2) # pJ/MAC
            if ii_a == 0:
                etype = f'E$_{{in\_regs\_update}}$'
            else:
                etype = None
            bar = ax1.bar(x_pos+width/2*ii_pos, values, width, label=etype, bottom = base, color = colors[5], edgecolor='black')
            if ii_a == 0:
                line_ax1.append(bar)
            base += values

            values = df.energy_mac_total.to_numpy()/(nb_ops/2) # pJ/MAC
            if ii_a == 0:
                etype = f'E$_{{mac}}$ (AIMC)'
            else:
                etype = f'E$_{{mac}}$ (DIMC)'
            bar = ax1.bar(x_pos+width/2*ii_pos, values, width, label=etype, bottom = base, color = colors[ii_a+6], edgecolor='black')
            line_ax1.append(bar)

        ax1.set_xticks(x_pos)
        if key == 'imc':
            r_angle = 45
        else:
            r_angle = 0
        ax1.set_xticklabels(labels, rotation=r_angle)
        ax1.set_title(f'{ltype} layer')
        #plt.ylabel(f'energy breakdown (pJ@{dfr.iloc[0].workload})')
        #ax1.set_ylabel(f'{ltype} layer')

        if ax2_enable == 'on':
            ax2 = ax1.twinx()
            #ax2.set_ylabel(f"I/O reuse percentage",color='black')
            ax2.tick_params(axis='y',labelcolor='black')
            line_ax2 = []
            for ii_a, a in enumerate(['aimc_max','dimc']):
                df = dfr[dfr.accelerator_type==a]
                x_pos = np.arange(len(labels))
                if key == 'imc':
                    df = df.sort_values(by='rows', ascending=True)
                    labels = df.imc.to_numpy() # x label
                elif key == 'cores':
                    df = df.sort_values(by='rows', ascending=False) # sort in an ascending order of 'cores'
                    labels = df.cores.to_numpy() # x label
                elif key == 'm':
                    df = df.sort_values(by='rows')
                    labels = df.m.to_numpy() # x label
                else:
                    breakpoint() # key incorrect, debug required
                #values = np.array([x for sublist in df.macro_utilization.to_list() for x in sublist])
                if ii_a == 0:
                    ii_pos = -1
                else:
                    ii_pos = 1
                if ii_a == 0:
                    linestyle = ':'
                else:
                    linestyle = '--'
                imc_marker = ['o','^']
                if ii_a == 0: # plot in/out reuse for both AIMC and DIMC
                    values = df.input_reuse.to_numpy()
                    line = ax2.plot(x_pos, values, linestyle='-', color = colors_twin[0], marker=imc_marker[ii_a], markeredgewidth=1.0, markeredgecolor='black', label=f'input unrolling ratio')
                    line_ax2.append(line[0])
                    values = df.output_reuse.to_numpy()
                    line = ax2.plot(x_pos, values, linestyle='-', color = colors_twin[1], marker=imc_marker[ii_a], markeredgewidth=1.0, markeredgecolor='black', label=f'output unrolling ratio')
                    line_ax2.append(line[0])
                    # for including macro utilization
                    # values = np.array([x for sublist in df.macro_utilization.to_list() for x in sublist])
                    # line = ax2.plot(x_pos, values, linestyle='-', color = colors_twin[2], marker=imc_marker[ii_a], markeredgewidth=1.0, markeredgecolor='black', label=f'macro utilization')
                    # line_ax2.append(line[0])
                    ax2.set_ylim(0,1.1)
                    ax2.set_yticks([y/10 for y in range(0,11,2)])
            lines = line_ax1 + line_ax2
        else:
            lines = line_ax1
        #ax1.set_axisbelow(True)
        #ax1.grid(which='both')
        #ax2.set_axisbelow(True)
        #ax2.grid(which='both')
        return lines

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    if key == 'imc':
        fig.text(0.5,0,f'IMC size (#rows: D$_i$, #columns: D$_o\\cdot$B$_w$)',ha='center') # x label
    elif key == 'cores':
        fig.text(0.5,0,f'#cores',ha='center') # x label
    else: # key == 'm'
        fig.text(0.5,0,f'M',ha='center') # x label
    fig.text(0.02,0.5,'Energy breakdown [pJ/MAC]',va='center', rotation='vertical') # y label
    fig.text(0.98,0.5,'in/out spatial unrolling ratio',va='center', rotation='vertical') # y label
    # dense layer
    workload, layer_idx, nb_ops = ['ae'], 0, 163840 # C=640, K=128
    lines = plot_e_breakdown(workload, layer_idx, nb_ops, axs[0][0], ltype='fc', key=key)
    # pw layer
    #workload, layer_idx, nb_ops = ['ds-cnn'], 2, 1024000 # C=K=64, OX/OY=25/5
    workload, layer_idx, nb_ops = ['mobilenet'], 10, 1179648 # C=K=64, OX=OY=12
    plot_e_breakdown(workload, layer_idx, nb_ops, axs[0][1], ltype='pw', key=key)
    # dw layer
    workload, layer_idx, nb_ops = ['ds-cnn'], 1, 144000 # G=64, FX=FY=3, OX/OY=25/5
    plot_e_breakdown(workload, layer_idx, nb_ops, axs[1][0], ltype='dw', key=key)
    ## conv layer
    #workload, layer_idx, nb_ops = ['ds-cnn'], 0, 640000 # K/C=64/1, FX/FY=10/4, OX/OY=25/5
    workload, layer_idx, nb_ops = ['resnet8'], 1, 4718592 # K/C=16/16, FX/FY=3/3, OX/OY=32/32
    plot_e_breakdown(workload, layer_idx, nb_ops, axs[1][1], ltype='conv', key=key)
    # Set the labels and title
    #plt.title(f'Energy breakdown (different {key} @ {dfr.iloc[0].workload})')
    # Add legend
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc='upper center', ncol=6, frameon=False)
    # Show the chart
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    workloads = ['ds-cnn', 'ae', 'mobilenet', 'resnet8'] # onnx folder name list
    workload_eval_multi_processing(workloads, mp=True, clr=False) # zigzag simulation (mp=True: multi-process; False: single-process)


