import pickle
import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import sim_auto
import peak_eval

def layer_info_read(workloads, layer_idx, nb_ops):
    ''' workloads: types of workload '''
    ''' layer_idx: which layer to extract info '''
    ''' nb_ops: used for calculating tops '''
    data_vals = []
    #ops_workloads = {'ae': 532512, 'ds-cnn': 5609536, 'mobilenet': 15907840, 'resnet8': 25302272}
    for ii_a, a in enumerate(['aimc_max','dimc']):
        for ii_c, workload in enumerate(workloads):
            directory = f'outputs/{a}/{workload}/'
            # read imc data
            for filename in os.listdir(directory):
                f = os.path.join(directory, filename)
                if os.path.isfile(f):
                    if f.split('.')[-1] == 'pkl':
                        with open(f,'rb') as infile:
                            data = pickle.load(infile)
                        if len(data[1]) == 0: # no output from zigzag
                            print(f'[WARNING] no output detected for some cases')
                            continue
                        dx = data[0]
                        cme_list = [v for k,v in data[1].items()]
                        if len(cme_list)==1:
                            cme_list = [cme_list[0]] # read info if only one layer is detected
                        else:
                            cme_list = [cme_list[layer_idx]] # only read info for specific layer
                        en_total = sum([v.energy_total for v in cme_list])
                        lat_total = sum([v.latency_total0 for v in cme_list])
                        lat_ideal_total = sum([v.ideal_temporal_cycle for v in cme_list])
                        lat_stall_total = lat_total - lat_ideal_total # about = weight_loading_cycles
                        dx['workload'] = workload
                        dx['ops'] = nb_ops # number of operations
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
                        dx['area_total'] = dx['area_imc'] + dx['area_cache'] #mm2
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

                        macro_size = dx['rows']*dx['cols']*int(dx['cores']) # bit
                        used_size = []
                        utilization = []
                        for ii_d, d in enumerate(cme_list):
                            unroll = d.spatial_mapping.unroll_size_total['O']
                            tm_unroll = d.temporal_mapping['O']
                            Ou = unroll[0] # Cu*FXu*FYu
                            Iu = unroll[1] # Ku
                            Otm = 1 # Ctm, FXtm, FYtm
                            Itm = 1 # Ktm
                            for ii_e, e in enumerate(tm_unroll): # read out tm unroll for C, FX, FY, K
                                if len(e) == 0: # empty at this mem level
                                    continue
                                for ii_g, g in enumerate(e):
                                    if g[0] in ['C', 'FX', 'FY']:
                                        Otm *= g[1]
                                    if g[0] in ['K']:
                                        Itm *= g[1]

                            input_reuse = Iu/(Iu*Itm) # percentage of spatial input unrolling
                            output_reuse = Ou/(Ou*Otm) # percentage of spatial output unrolling
                            # calculate weight spatial mapping when M > 1
                            weight_reuse = 1
                            if 'W' in d.temporal_mapping: # cal used #rows when M > 1
                                for f in d.temporal_mapping['W'][0]:
                                    if f[0] == 'FX' or f[0] == 'FY' or f[0] == 'K' or f[0] == 'C':
                                        ''' all possible r loop for weight '''
                                        weight_reuse *= f[1]
                            elif 'B' in d.temporal_mapping:
                                for f in d.temporal_mapping['B'][0]:
                                    if f[0] == 'FX' or f[0] == 'FY' or f[0] == 'K' or f[0] == 'C':
                                        ''' all possible r loop for weight '''
                                        weight_reuse *= f[1]
                            else:
                                breakpoint() # there should be no other case. debug required.
                            act_pres = int(dx['act_pres']) #  activation precision
                            used_size.append(Ou*Iu*weight_reuse*act_pres)
                            utilization.append(Ou*Iu*weight_reuse*act_pres/macro_size)
                        dx['input_reuse'] = input_reuse
                        dx['output_reuse'] = output_reuse
                        dx['macro_utilization'] = utilization
                        data_vals.append(dx)
    
     
    df = pd.DataFrame(data_vals)
    return df

def plt_latency_breakdown(dfr):
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    dfr['imc'] = dfr.rows.astype(str)+f"$\\times$"+dfr.cols.astype(str)
    dfr = dfr[(dfr.act_pres==8) & ( ((dfr.input_precision==2)&(dfr.accelerator_type=='aimc_max')) | ((dfr.input_precision==1)&(dfr.accelerator_type=='dimc')) ) & (dfr.rows == dfr.cols)]
    dfr = dfr.sort_values(by=['rows'])

    width = 0.35
    for ii_a, a in enumerate(['aimc_max','dimc']):
        if ii_a == 0:
            ii_pos = -1
        else:
            ii_pos = 1

        df = dfr[dfr.accelerator_type==a]
        labels = df.imc.to_numpy() # x label
        # Create positions for the bars on the x-axis
        x_pos = np.arange(len(labels))
        # Plot the bars
        base = [0 for x in range(0, len(labels))]
        values = df.latency_stall_total.to_numpy()
        if ii_a == 0:
            etype = f'Lat$_{{stall}}$'
        else:
            etype = None
        plt.bar(x_pos+width/2*ii_pos, values, width, label=etype, bottom = base, color = colors[0])
        base += values
        values = df.latency_ideal_total.to_numpy()
        if ii_a == 0:
            etype = f'Lat$_{{mac}} (AIMC)$'
        else:
            etype = f'Lat$_{{mac}} (DIMC)$'
        plt.bar(x_pos+width/2*ii_pos, values, width, label=etype, bottom = base, color = colors[ii_a+1])

    # Set the labels and title
    plt.xticks(x_pos, labels)
    plt.xlabel(f'IMC size')
    plt.ylabel(f'Latency breakdown (cycles@{dfr.iloc[0].workload})')
    plt.title(f'Latency breakdown @ different IMC size (@{dfr.iloc[0].workload})')
    # Add legend
    plt.legend()
    # Show the chart
    plt.show()

def plt_area_breakdown(dfr):
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    dfr['imc'] = dfr.rows.astype(str)+f"$\\times$"+dfr.cols.astype(str)
    dfr = dfr[(dfr.act_pres==8) & ( ((dfr.input_precision==2)&(dfr.accelerator_type=='aimc_max')) | ((dfr.input_precision==1)&(dfr.accelerator_type=='dimc')) ) & (dfr.rows == dfr.cols)]
    dfr = dfr.sort_values(by=['rows'])

    width = 0.35
    for ii_a, a in enumerate(['aimc_max','dimc']):
        if ii_a == 0:
            ii_pos = -1
        else:
            ii_pos = 1

        df = dfr[dfr.accelerator_type==a]
        labels = df.imc.to_numpy() # x label
        # Create positions for the bars on the x-axis
        x_pos = np.arange(len(labels))
        # Plot the bars
        base = [0 for x in range(0, len(labels))]
        values = df.area_cache.to_numpy()
        if ii_a == 0:
            etype = f'Area$_{{cache}}$'
        else:
            etype = None
        plt.bar(x_pos+width/2*ii_pos, values, width, label=etype, bottom = base, color = colors[0])
        base += values
        values = df.area_imc.to_numpy()
        if ii_a == 0:
            etype = f'Area$_{{macro}} (AIMC)$'
        else:
            etype = f'Area$_{{macro}} (DIMC)$'
        plt.bar(x_pos+width/2*ii_pos, values, width, label=etype, bottom = base, color = colors[ii_a+1])

    # Set the labels and title
    plt.xticks(x_pos, labels)
    plt.xlabel(f'IMC size')
    plt.ylabel(f'Area breakdown (mm$^2$@{dfr.iloc[0].workload})')
    plt.title(f'Area breakdown @ different IMC size (@{dfr.iloc[0].workload})')
    # Add legend
    plt.legend()
    # Show the chart
    plt.show()

def energy_breakdown_across_layers(workload, user_rows=32):
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    dfr = sim_auto.workload_output_read([workload])
    dfr['imc'] = dfr.rows.astype(str)+f"$\\times$"+dfr.cols.astype(str)
    dfr = dfr[((dfr.accelerator_type=='aimc_max')&(dfr.input_precision==2)) | ((dfr.accelerator_type=='dimc')&(dfr.input_precision==1))]
    dfr = dfr[(dfr.cores=='1')&(dfr.m=='1')]
    dfr = dfr[(dfr.cols==user_rows)] # only plot for specific case
    #dfr = dfr[(dfr.rows==user_rows)] # only plot for specific case
    imcs = ['aimc_max','dimc']
    print(dfr)
    print(dfr.m)
    print(dfr.topsw_workload)
    print(dfr.tops_workload)
    print(dfr.topsmm2_workload)
    print(dfr.latency_total)
    print(dfr.latency_stall_total)
    print(dfr.latency_ideal_total)

    ''' plot energy breakdown of each layer for one aimc and one dimc configuration'''
    result = []
    for imc in imcs:
        dfx = dfr[dfr.accelerator_type==imc]
        e_breakdown = [[],[],[],[],[],[],[]]
        for ii_a, a in enumerate(dfx.energy_mem_breakdown.to_numpy()[0]):
            e_breakdown[6].append( dfx.cme.to_numpy()[0][ii_a].MAC_energy )
            for ii_k ,k in enumerate(a.keys()):
                if ii_k == 0: # output energy
                    e_breakdown[4].append( a[k][0] )
                    try:
                        e_breakdown[5].append( a[k][1] )
                    except: # mem level is removed
                        e_breakdown[5].append( 0 )
                elif ii_k == 1: # weight energy
                    e_breakdown[0].append( a[k][0] )
                    try:
                        e_breakdown[1].append( a[k][1] )
                    except:
                        e_breakdown[1].append( 0 )
                elif ii_k == 2:
                    e_breakdown[2].append( a[k][0] )
                    try:
                        e_breakdown[3].append( a[k][1] )
                    except:
                        e_breakdown[3].append( 0 )
                else:
                    breakpoint()
        result.append(e_breakdown)

    # Create the labels
    labels = [f'layer\n{idx}' for idx in range(0, len(e_breakdown[0]))]
    legends = [f'E$_{{WeightUpdate}}$', f'E$_{{WeightDram}}$', f'E$_{{InReg}}$', f'E$_{{InCache}}$',f'E$_{{OutReg}}$', f'E$_{{OutCache}}$', f'E$_{{MAC}}$']
    x_pos = np.arange(len(labels))
    # Plot the bars
    width = 0.2
    for ii_imc, imc in enumerate(imcs):
        base = [0 for x in range(0, len(labels))]
        if len(imcs)>1 and ii_imc==0:
            ii_pos = -1
        elif len(imcs)>1 and ii_imc==1:
            ii_pos = 1
        else:
            ii_pos = 0

        for ii_a, a in enumerate(result[ii_imc]):
            if ii_imc == 0:
                if ii_a == 6:
                    plt.bar(x_pos+width/2*ii_pos, a, width, label=f'E$_{{MAC (AIMC)}}$', bottom = base, color=colors[0])
                else:
                    plt.bar(x_pos+width/2*ii_pos, a, width, label=legends[ii_a], bottom = base, color=colors[ii_a+1])
            else:
                if ii_a == 6:
                    plt.bar(x_pos+width/2*ii_pos, a, width, label=f'E$_{{MAC (DIMC)}}$', bottom = base, color=colors[ii_a+2])
                else:
                    plt.bar(x_pos+width/2*ii_pos, a, width, bottom = base, color=colors[ii_a+1])

            for ii_x,x in enumerate(range(0, len(a))):
                base[ii_x] += a[ii_x]

    # Set the labels and title
    plt.xticks(x_pos, labels)
    plt.xlabel(f'Layer idx')
    plt.ylabel(f'energy breakdown of each layer (pJ@{dfr.iloc[0].workload})')
    plt.title(f'Energy breakdown of layers (@{dfr.iloc[0].workload})')
    # Add legend
    plt.legend()
    # Show the chart
    plt.show()
    #breakpoint()


if __name__ == '__main__':
    ''' FOR DEBUGGING '''
    ## dense layer
    #workload, layer_idx, nb_ops = ['ae'], 0, 163840 # C=640, K=128
    #workload, layer_idx, nb_ops = ['ae'], 1, 32768 # C=128, K=128

    ## pointwise layer
    #workload, layer_idx, nb_ops = ['ds-cnn'], 2, 1024000 # C=K=64, OX/OY=25/5
    #workload, layer_idx, nb_ops = ['mobilenet'], 14, 1179648 # C=K=128, OX=OY=6
    #workload, layer_idx, nb_ops = ['mobilenet'], 8, 589824 # C/K=32/64, OX=OY=12

    # dw layer
    #workload, layer_idx, nb_ops = ['ds-cnn'], 1, 144000 # G=64, FX=FY=3, OX/OY=25/5
    #workload, layer_idx, nb_ops = ['mobilenet'], 1, 331776 # G=8, FX=FY=3, OX/OY=48/48

    # conv layer
    #workload, layer_idx, nb_ops = ['ds-cnn'], 0, 640000 # K/C=64/1, FX/FY=10/4, OX/OY=25/5
    #workload, layer_idx, nb_ops = ['resnet8'], 7, 4718592 # K/C=64/64, FX/FY=3/3, OX/OY=8/8
    #workload, layer_idx, nb_ops = ['mobilenet'], 0, 995328 # K/C=8/3, FX/FY=3/3, OX/OY=48/48
    workload, layer_idx, nb_ops = ['resnet8'], 4, 4718592 # K/C=32/32, FX/FY=3/3, OX/OY=16/16

    ''' FILTER TO ASSIST DEBUGGING '''
    dfr = layer_info_read(workload, layer_idx, nb_ops)
    dfr = dfr[((dfr.accelerator_type=='aimc_max')&(dfr.input_precision==2)) | ((dfr.accelerator_type=='dimc')&(dfr.input_precision==1))]
    dfr = dfr[(dfr.cores=='1')&(dfr.m=='1')].sort_values(by='rows')
    dfr['imc'] = dfr.rows.astype(str)+f"$\\times$"+dfr.cols.astype(str)
    #breakpoint()

    ''' function for plotting (not used in the paper) '''
    #energy_breakdown_across_layers('mobilenet', user_rows=128)
    #plt_latency_breakdown(dfr)
    #plt_area_breakdown(dfr)

    ''' functions for plotting (varied array size) in paper '''
    #sim_auto.workload_vs_system(en_layer='off',simulation='off', key='imc') # curves for peak topsw, topsmm2 (Fig. 7)
    #sim_auto.workload_vs_system(en_layer='on', key='imc',simulation='off') # curves for single layer (Fig. 10)
    #sim_auto.plt_multiple_energy_breakdown(key='imc') # energy breakdown of single layer (Fig. 11)
    #sim_auto.plt_multiple_latency_breakdown(key='imc') # latency breakdown of single layer (Fig. not used in the paper)
    #sim_auto.workload_vs_system(en_workload='on') # curves for entire workload (Fig. 12)


