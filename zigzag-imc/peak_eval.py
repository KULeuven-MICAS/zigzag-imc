import pickle
import math
import numpy as np
import pandas as pd
import plotly.express as px
import get_aimc_cost
import get_imc_cost
import matplotlib.pyplot as plt
import tqdm

def peak_eval(rows_list=None, cores_list=None, m_list=None, activation_precision_list=None, input_precision_list=None, imc_list=None):
    ''' evaluate peak system performance and peak macro performance '''
    '''
    :rows_list  : nb_rows of one macro (column = row in each round)
    :cores_list : nb_macros
    :m_list     : nb_cells sharing one input channel/sharing one multiplier
    :activation_precision_list  : operand precision list
    :input_precision_list       : B_cycle list
    :imc_list   : list of imc types
    '''
    ###########
    # setting #
    #activation_precision, weight_precision, output_precision = 4,4,4
    cacti_path = './cacti-master'
    unit_area_28nm = 0.614 # um2
    #unit_delay_28nm = 0.0374 # ns
    unit_delay_28nm = 0.0478 # ns
    unit_cap_28nm = 0.7 # fF
    vdd = 0.9 # V
    
    # exploration space
    if rows_list == None:
        u_rows_list = [2**x for x in range(5,11)]
    else:
        u_rows_list = rows_list
    if cores_list == None:
        u_cores_list = [1]
    else:
        u_cores_list = cores_list
    if activation_precision_list == None:
        u_act_list = [8]
    else:
        u_act_list = activation_precision_list
    if input_precision_list == None:
        u_input_list = [1,2]
    else:
        u_input_list = input_precision_list
    if m_list == None:
        u_m_list = [1]
    else:
        u_m_list = m_list
    if imc_list == None:
        u_imc_list = ['aimc_max', 'dimc']
    else:
        u_imc_list = imc_list

    data_vals=[]
    for ii_a, total_rows in enumerate(u_rows_list):
        for total_cols in [total_rows//u_m_list[0]]:
            for number_of_cores in tqdm.tqdm([1]):
                for cells_per_mult in u_m_list:
                    for activation_precision in u_act_list:
                        weight_precision = activation_precision
                        output_precision = activation_precision
                        for imc_type in u_imc_list:
                            for input_precision in [1 if imc_type=='dimc' else 2]:
                            #for input_precision in u_input_list:
                                #print(number_of_cores,cells_per_mult,input_precision,imc_type)
                                if imc_type == 'aimc_max':
                                    max_adc_resolution = 'dynamic' # dynamic: input_res+0.5log2(input_channel); else (fixed): input_pres
                                elif imc_type == 'aimc_min':
                                    max_adc_resolution = 'fixed'
                                print(f'setting, rows: {total_rows}, cols: {total_cols}, cores: {number_of_cores}, M: {cells_per_mult}, Bi: {input_precision}, imc: {imc_type}')

                                rows = total_rows*cells_per_mult# rows per macro
                                input_channel = rows/cells_per_mult # in_chl per macro
                                if input_channel < 1:
                                    continue # skip these no-sense case
                                cols = total_cols# cols per macro
                                output_channel = cols/weight_precision # out_chl per macro
                                if cols < 8 or rows < 32 or cols*rows/8<64:
                                    continue #cacti doesn't support

                                dx = {'imc_type': imc_type}
                                dx['act_precision'] = str(activation_precision)
                                dx['input_precision'] = str(input_precision)
                                dx['rows'] = total_rows
                                dx['cols'] = total_cols
                                dx['cores'] = str(number_of_cores)
                                dx['m'] = str(cells_per_mult)
                                if imc_type in ['aimc_max','aimc_min']:
                                    ''' for each macro, we have '''
                                    area_macro_aimc, tclk_macro_aimc, tops_macro_aimc, topsw_macro_aimc, topsmm2_macro_aimc, mem_density, aimc_dict_macro = get_aimc_cost.get_aimc_cost(cacti_path, activation_precision, input_precision, weight_precision, output_precision, input_channel, output_channel, input_channel*cells_per_mult, 1, 0, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd, max_adc_resolution, eval_level='macro',cores=number_of_cores)
                                    area_system_aimc, tclk_system_aimc, tops_system_aimc, topsw_system_aimc, topsmm2_system_aimc, mem_density, aimc_dict_system = get_aimc_cost.get_aimc_cost(cacti_path, activation_precision, input_precision, weight_precision, output_precision, input_channel, output_channel, input_channel*cells_per_mult, 1, 0, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd, max_adc_resolution, eval_level='system',cores=number_of_cores)

                                    ''' for all macros, we have '''
                                    dx['dac_precision'] = aimc_dict_macro['dac_resolution']
                                    dx['adc_precision'] = aimc_dict_macro['adc_resolution']
                                    dx['adc_energy_pj'] = aimc_dict_macro['peak_energy_breakdown_adcs (pJ)']
                                    dx['dac_energy_pj'] = aimc_dict_macro['peak_energy_breakdown_dacs (pJ)']
                                    dx['energy_rego2cache_pj'] = aimc_dict_system['peak_energy_breakdown_output_regs_2_cache (pJ)']
                                    dx['energy_cache2regi_pj'] = aimc_dict_system['peak_energy_breakdown_cache_2_input_regs (pJ)']
                                    dx['energy_regi_load_pj'] = aimc_dict_system['peak_energy_breakdown_input_regs (pJ)']
                                    dx['energy_rego_load_pj'] = aimc_dict_system['peak_energy_breakdown_output_regs (pJ)']
                                    dx['energy_bank'] = aimc_dict_system['peak_energy_breakdown_bank (pJ)']
                                    dx['energy_mults'] = aimc_dict_system['peak_energy_breakdown_mults (pJ)']
                                    dx['energy_adders'] = aimc_dict_system['peak_energy_breakdown_adder_tree (pJ)']
                                    dx['energy_accumulators'] = aimc_dict_system['peak_energy_breakdown_accumulator (pJ)']+aimc_dict_system['peak_energy_breakdown_regs_accumulator (pJ)']
                                    dx['area_adc'] = aimc_dict_system['area_breakdown_adc (mm2)']
                                    dx['area_dac'] = aimc_dict_system['area_breakdown_dac (mm2)']
                                    dx['area_bank'] = aimc_dict_system['area_breakdown_bank (mm2)']
                                    dx['area_mults'] = aimc_dict_system['area_breakdown_mults (mm2)']
                                    dx['area_adders'] = aimc_dict_system['area_breakdown_adder_tree (mm2)']
                                    dx['area_accumulators'] = aimc_dict_system['area_breakdown_accumulator (mm2)']+aimc_dict_system['area_breakdown_regs_accumulator (mm2)']
                                    dx['area_cache'] = aimc_dict_system['area_cache (mm2)']
                                    dx['area_regi'] = aimc_dict_system['area_input_regs (mm2)']
                                    dx['area_rego'] = aimc_dict_system['area_output_regs (mm2)']
                                    dx['area_macro'] = area_macro_aimc
                                    dx['tclk_macro'] = tclk_macro_aimc
                                    dx['tops_macro'] = tops_macro_aimc
                                    dx['topsw_macro'] = topsw_macro_aimc
                                    dx['topsmm2_macro'] = topsmm2_macro_aimc
                                    dx['area_system'] = area_system_aimc
                                    dx['tclk_system'] = tclk_system_aimc                    # actually equal to tclk_macro_aimc
                                    dx['tops_system'] = tops_system_aimc
                                    dx['topsw_system'] = topsw_system_aimc
                                    dx['topsmm2_system'] = topsmm2_system_aimc
                                    dx['aimc_dict_macro'] = aimc_dict_macro
                                    dx['aimc_dict_system'] = aimc_dict_system
                                    data_vals.append(dx)
                                elif math.log2(input_channel)%1 != 0:
                                    continue # not power of 2, not supported by adder tree in DIMC
                                elif imc_type == 'dimc': # DIMC
                                    ''' for each macro, we have '''
                                    area_macro_dimc, tclk_macro_dimc, tops_macro_dimc, topsw_macro_dimc, topsmm2_macro_dimc, mem_density, dimc_dict_macro = get_imc_cost.get_imc_cost(cacti_path, activation_precision, input_precision, weight_precision, output_precision, input_channel, output_channel, rows, 1, 0, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd, eval_level='macro',cores=number_of_cores)
                                    area_system_dimc, tclk_system_dimc, tops_system_dimc, topsw_system_dimc, topsmm2_system_dimc, mem_density, dimc_dict_system = get_imc_cost.get_imc_cost(cacti_path, activation_precision, input_precision, weight_precision, output_precision, input_channel, output_channel, rows, 1, 0, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd, eval_level='system',cores=number_of_cores)

                                    ''' for all macros, we have '''
                                    dx['energy_rego2cache_pj'] = dimc_dict_system['peak_energy_breakdown_output_regs_2_cache (pJ)']
                                    dx['energy_cache2regi_pj'] = dimc_dict_system['peak_energy_breakdown_cache_2_input_regs (pJ)']
                                    dx['energy_regi_load_pj'] = dimc_dict_system['peak_energy_breakdown_input_regs (pJ)']
                                    dx['energy_rego_load_pj'] = dimc_dict_system['peak_energy_breakdown_output_regs (pJ)']
                                    dx['energy_bank'] = dimc_dict_system['peak_energy_breakdown_bank (pJ)']
                                    dx['energy_mults'] = dimc_dict_system['peak_energy_breakdown_mults (pJ)']
                                    dx['energy_adders'] = dimc_dict_system['peak_energy_breakdown_adder_tree (pJ)']
                                    dx['energy_accumulators'] = dimc_dict_system['peak_energy_breakdown_accumulator (pJ)']+dimc_dict_system['peak_energy_breakdown_regs_accumulator (pJ)']
                                    dx['area_bank'] = dimc_dict_system['area_breakdown_bank (mm2)']
                                    dx['area_mults'] = dimc_dict_system['area_breakdown_mults (mm2)']
                                    dx['area_adders'] = dimc_dict_system['area_breakdown_adder_tree (mm2)']
                                    dx['area_accumulators'] = dimc_dict_system['area_breakdown_accumulator (mm2)']+dimc_dict_system['area_breakdown_regs_accumulator (mm2)']
                                    dx['area_cache'] = dimc_dict_system['area_cache (mm2)']
                                    dx['area_regi'] = dimc_dict_system['area_input_regs (mm2)']
                                    dx['area_rego'] = dimc_dict_system['area_output_regs (mm2)']
                                    dx['area_macro'] = area_macro_dimc
                                    dx['tclk_macro'] = tclk_macro_dimc
                                    dx['tops_macro'] = tops_macro_dimc
                                    dx['topsw_macro'] = topsw_macro_dimc
                                    dx['topsmm2_macro'] = topsmm2_macro_dimc
                                    dx['area_system'] = area_system_dimc
                                    dx['tclk_system'] = tclk_system_dimc                    # actually equal to tclk_macro_dimc
                                    dx['tops_system'] = tops_system_dimc
                                    dx['topsw_system'] = topsw_system_dimc
                                    dx['topsmm2_system'] = topsmm2_system_dimc
                                    dx['dimc_dict_macro'] = dimc_dict_macro
                                    dx['dimc_dict_system'] = dimc_dict_system
                                    data_vals.append(dx)
                                else:
                                    print(f'[WARNING] undefined imc_type: {imc_type}')
    df = pd.DataFrame(data_vals)
    df.to_pickle('peak_eval_256kB.pickle')

def breakdown_vs_imc_size():
    '''
    #cores = 1, M=1
    '''
    #colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'] # default color plattee
    colors = [u'#633E7A', u'#ff7f0e', u'#CCD5AE', u'#FEFAE0', u'#27E1C1', u'#BDE0FE', u'#CDB4DB', u'#FAD2E1']
    df =  pd.read_pickle('peak_eval_256kB.pickle')
    df['imc'] = df.rows.astype(str)+f"$\\times$"+df.cols.astype(str)
    input_precisions = ['2','1'] # 2@aimc, 1@dimc
    energys = [[],[]]
    areas = [[],[]]
    tclks = [[], []]
    bar_width = 0.35
    def fetch_delay(dfx, a, dtype):
        xx = []
        for x in range(0, len(dfx.imc.to_numpy())):
            if a=='aimc_max':
                xx.append( dfx.aimc_dict_system.to_numpy()[x][dtype] )
            else:
                xx.append( dfx.dimc_dict_system.to_numpy()[x][dtype] )
        return np.array(xx)
    for ii_a, a in enumerate(['aimc_max','dimc']):
        dfx = df[(df.imc_type==a)&(df.input_precision==input_precisions[ii_a])]
        labels = dfx.imc.to_numpy()
        energys[ii_a].append( fetch_delay(dfx, a, 'peak_energy_breakdown_cache (fJ/MAC)'))
        energys[ii_a].append( fetch_delay(dfx, a, 'peak_energy_breakdown_regs (fJ/MAC)'))
        energys[ii_a].append( fetch_delay(dfx, a, 'peak_energy_breakdown_mults_mac (fJ/MAC)'))
        energys[ii_a].append( fetch_delay(dfx, a, 'peak_energy_breakdown_adder_tree_mac (fJ/MAC)'))
        energys[ii_a].append( fetch_delay(dfx, a, 'peak_energy_breakdown_accumulator_mac (fJ/MAC)'))
        energys[ii_a].append( fetch_delay(dfx, a, 'peak_energy_breakdown_bank_mac (fJ/MAC)'))
        areas[ii_a].append( dfx.area_cache.to_numpy() )
        #areas[ii_a].append( dfx.area_regi.to_numpy() )
        #areas[ii_a].append( dfx.area_rego.to_numpy() )
        areas[ii_a].append( dfx.area_regi.to_numpy() + dfx.area_rego.to_numpy() )
        areas[ii_a].append( dfx.area_mults.to_numpy() )
        areas[ii_a].append( dfx.area_adders.to_numpy() )
        areas[ii_a].append( dfx.area_accumulators.to_numpy() )
        areas[ii_a].append( dfx.area_bank.to_numpy() )
        tclks[ii_a].append( fetch_delay(dfx, a, 'tclk_breakdown_mults (ns)'))
        tclks[ii_a].append( fetch_delay(dfx, a, 'tclk_breakdown_adder_tree (ns)'))
        tclks[ii_a].append( fetch_delay(dfx, a, 'tclk_breakdown_accumulator (ns)'))
        tclks[ii_a].append( fetch_delay(dfx, a, 'tclk_breakdown_cells (ns)'))
        if a=='aimc_max':
            energys[ii_a].append( fetch_delay(dfx, a, 'peak_energy_breakdown_adcs_mac (fJ/MAC)'))
            energys[ii_a].append( fetch_delay(dfx, a, 'peak_energy_breakdown_dacs_mac (fJ/MAC)'))
            areas[ii_a].append( dfx.area_adc.to_numpy() )
            areas[ii_a].append( dfx.area_dac.to_numpy() )
            tclks[ii_a].append( fetch_delay(dfx, a, 'tclk_breakdown_adc (ns)'))
            tclks[ii_a].append( fetch_delay(dfx, a, 'tclk_breakdown_dac (ns)'))

    fig, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].set_axisbelow(True)
    axs[1].set_axisbelow(True)
    axs[2].set_axisbelow(True)
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()

    area_ax = axs[2]
    tclk_ax = axs[1]
    energy_ax = axs[0]
    fig.text(0.5,0,f'IMC size (#rows: D$_i$, #columns: D$_o\\cdot$B$_w$)',ha='center', fontsize=12) # x label
    area_ax.set_ylabel(f'Area breakdown [mm$^2$]', fontsize=12)
    x_pos = np.arange(len(labels))
    area_ax.set_xticks(x_pos)
    area_ax.set_xticklabels(labels, rotation=45)
    #area_ax.tick_params(axis='x', labelsize=10)
    area_labels = ['cache (256 KB)','in/out registers','multipliers','adder trees','accumulators','cell_array','ADCs', 'DACs']
    for ii_a, a in enumerate(areas):
        base = [0 for x in range(0, len(labels))]
        if ii_a == 0: #aimc
            ii_pos = -1
            en_label = True
        else: #dimc
            ii_pos = 1
            en_label = False
        for ii_c, c in enumerate(a):
            if en_label == True: # only plot legend once
                area_ax.bar(x_pos+bar_width/2*ii_pos, c, bar_width, label=area_labels[ii_c], bottom=base, color=colors[ii_c], edgecolor='black')
            else:
                area_ax.bar(x_pos+bar_width/2*ii_pos, c, bar_width, bottom=base, color=colors[ii_c], edgecolor='black')
            base += c

    fig.legend(loc='upper center', ncol=4, frameon=False)
    tclk_ax.set_ylabel(f'Tclk breakdown [ns]', fontsize=12)
    tclk_ax.set_xticks(x_pos)
    tclk_ax.set_xticklabels(labels, rotation=45)
    for ii_a, a in enumerate(tclks):
        base = [0 for x in range(0, len(labels))]
        if ii_a == 0: #aimc
            ii_pos = -1
            en_label = True
        else: #dimc
            ii_pos = 1
            en_label = False
        for ii_c, c in enumerate(a):
            if en_label == True:
                tclk_ax.bar(x_pos+bar_width/2*ii_pos, c, bar_width, label=area_labels[ii_c], bottom=base, color=colors[ii_c+2], edgecolor='black')
            else:
                tclk_ax.bar(x_pos+bar_width/2*ii_pos, c, bar_width, bottom=base, color=colors[ii_c+2], edgecolor='black')
            base += c

    energy_ax.set_ylabel(f'Energy breakdown [fJ/MAC]', fontsize=12)
    energy_ax.set_xticks(x_pos)
    energy_ax.set_xticklabels(labels, rotation=45)
    for ii_a, a in enumerate(energys):
        base = [0 for x in range(0, len(labels))]
        if ii_a == 0: #aimc
            ii_pos = -1
            en_label = True
        else: #dimc
            ii_pos = 1
            en_label = False
        for ii_c, c in enumerate(a):
            if en_label == True:
                energy_ax.bar(x_pos+bar_width/2*ii_pos, c, bar_width, label=area_labels[ii_c], bottom=base, color=colors[ii_c], edgecolor='black')
            else:
                energy_ax.bar(x_pos+bar_width/2*ii_pos, c, bar_width, bottom=base, color=colors[ii_c], edgecolor='black')
            base += c

    area_ax.set_yticks( np.arange(0,8,1) )
    tclk_ax.set_yticks( np.arange(0,60,5) )
    energy_ax.set_yticks( np.arange(0,1100,100) )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #peak_eval() # get pickle
    breakdown_vs_imc_size()
    exit()

