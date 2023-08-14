from dimc_definition import *
import random
from get_cacti_cost import *
import math
import pdb
import logging

def get_imc_cost(cacti_path, activation_precision, input_precision, weight_precision, output_precision, input_channel, output_channel, rows, input_toggle_rate, weight_sparsity, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd, eval_level='macro', cores=1, ini_hash=0):
    '''
    get area cost for a specific dimc setting (include regs_input and regs_output)
    :param cacti_path:              path where to run cacti
    :param activation_precision:    activation bit precision (unit: bit)
    :param  input_precision:        how many bits per input channel (unit: bit)
    :param weight_precision:        weight bit precision (unit: bit)
    :param output_precision:        output bit precision (unit: bit) (affect the size of output registers)
    :param input_channel:           the number of input channel
    :param output_channel:          the number of output channel
    :param rows:                    the number of rows of SRAM cell array
    :unit_area_28nm:                the area cost of single NAND2 gate for 28nm process
    :unit_delay_28nm:               the delay cost of single NAND2 gate for 28nm process
    :unit_cap_28nm:                 the capancitance of single NAND2 gate for 28nm process
    :vdd:                           the nominal voltage for 28nm process
    :cols:                          the number of columns of SRAM cell array
    :col_mux:                       how many columns share one multiplier (>=1)
    :input_toggle_rate:             input toggle rate (0-1)
    :weight_sparsity:               weight sparsity on word level (bit sparsity inside a single weight is not taken into account)
    :eval_level:                    evaluation level (system or macro). The peak energy and area is different depending on the level
    :cores:                         define how many cores when caluclating system peak-performance (no need to define if no care about system peak performance)
    :ini_hash:                      used to calculate hash when call get_cacti_cost. Useful when running scripts in multiprocess.
    '''
    
    #unit_area_28nm = 0.614
    #unit_delay_28nm = 0.0374
    #unit_cap_28nm = 0.7
    #vdd=0.9

############################# IMC MACRO #######################

    col_mux = 1 # (>1 is also supported)
    cols = weight_precision * output_channel * col_mux
    w_bw = cols
    sram_size = rows * cols
    input_bw = input_channel * activation_precision
    #input_bw = input_channel * input_precision
    output_bw = output_channel * output_precision
    
    hash_tuple = (activation_precision, input_precision, input_channel, output_channel, rows, input_toggle_rate, weight_sparsity, vdd, cores, ini_hash, random.randbytes(1))
    sram_access_time, sram_area, sram_r_cost, sram_w_cost = get_cacti_cost(cacti_path = cacti_path, mem_type = 'sram', mem_size_in_byte = sram_size/8, bw = w_bw, hd_hash=str(hash(hash_tuple)))
    
    dimc_macro = DimcMacro(activation_precision, weight_precision, input_precision, input_channel, output_channel, input_channel, output_channel, rows, cols, w_bw, sram_access_time, sram_w_cost, sram_area, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd)
    
    regs_input = MemoryInstance(name='regs_input', size=input_bw, r_bw=input_bw, w_bw=input_bw, r_cost=0, w_cost=input_bw*UnitDff(unit_area_28nm, unit_delay_28nm, unit_cap_28nm).calculate_cap() * vdd**2, area=input_bw*UnitDff(unit_area_28nm, unit_delay_28nm, unit_cap_28nm).calculate_area(), r_port=1, w_port=1, rw_port=0, latency=1)
    regs_output = MemoryInstance(name='regs_output', size=output_bw, r_bw=output_bw, w_bw=output_bw, r_cost=0, w_cost=output_bw*UnitDff(unit_area_28nm, unit_delay_28nm, unit_cap_28nm).calculate_cap() * vdd**2, area=output_bw*UnitDff(unit_area_28nm, unit_delay_28nm, unit_cap_28nm).calculate_area(), r_port=1, w_port=1, rw_port=0, latency=1)
    
    tclk = max(dimc_macro.calculate_computing_delay(), dimc_macro.calculate_w_delay()) # ns

    number_of_ops = 2 * dimc_macro.calculate_number_of_macs() * cores
    tops = number_of_ops / tclk / 1000 # peak tops

    if eval_level=='system': # used for peak system performance evaluation
        ''' get cacti cost for SRAM (cache for activation and output) '''
        mem_size = 256 * 1024 # unit: byte
        cache_bw = max(input_bw*cores, output_bw*cores)
        ac, cache_area, sram_r_cost, sram_w_cost = get_cacti_cost(cacti_path, 'sram', mem_size, cache_bw, hd_hash=str(hash(hash_tuple)))
        sram_r_cost = sram_r_cost*(10**6) / cache_bw # fJ/bit for SRAM
        sram_w_cost = sram_w_cost*(10**6) / cache_bw # fJ/bit for SRAM
        regs_in_cache = sram_r_cost * input_bw  # for each macro
        regs_out_cache = sram_w_cost * output_bw  # for each macro
        energy = (dimc_macro.calculate_computing_energy(input_toggle_rate, weight_sparsity)+(regs_input.w_cost+regs_output.w_cost + regs_in_cache + regs_out_cache)/(activation_precision/input_precision))*cores #fJ
        total_area = (dimc_macro.calculate_area() + regs_input.area + regs_output.area) * cores + cache_area
    else:
        cache_area = 0
        regs_in_cache = 0
        regs_out_cache = 0
        energy = dimc_macro.calculate_computing_energy(input_toggle_rate, weight_sparsity) * cores #fJ
        total_area = dimc_macro.calculate_area() * cores

    topsw = number_of_ops / energy * 1000

    topsmm2 = tops / total_area

    mem_density = total_area / sram_size # mm2/bit

    peak_dict = {
            'eval_level'    : eval_level,
            'total_area'    : total_area,
            'tclk (ns)'     : tclk,
            'tops'          : tops,
            'topsw'         : topsw,
            'topsmm2'       : topsmm2,
            'mem_density'   : mem_density,
            'area_macros (mm2)': dimc_macro.calculate_area()*cores,
            'area_breakdown_bank (mm2)':    dimc_macro.area_bank,
            'area_breakdown_mults (mm2)':   dimc_macro.area_mults,
            'area_breakdown_adder_tree (mm2)':   dimc_macro.area_adder_tree,
            'area_breakdown_accumulator (mm2)':   dimc_macro.area_accumulator,
            'area_breakdown_regs_accumulator (mm2)':   dimc_macro.area_regs_accumulator,
            'area_cache (mm2)'          :   cache_area,
            'area_input_regs (mm2)'     :   regs_input.area*cores,
            'area_output_regs (mm2)'    :   regs_output.area*cores,
            'tclk_breakdown_cells (ns)' :   dimc_macro.array_delay,
            'tclk_breakdown_mults (ns)' :   dimc_macro.mults.calculate_delay(),
            'tclk_breakdown_adder_tree (ns)': dimc_macro.adder_tree.calculate_delay(),
            'tclk_breakdown_accumulator (ns)':dimc_macro.accumulator.calculate_delay_msb(),
            'imc_weight_update_delay (ns)'  : dimc_macro.calculate_w_delay(),
            'imc_energy_under_peak_performance/cycle (pJ)': energy/1000, #pJ
            '#ops_under_peak_performance/cycle':    number_of_ops,
            'peak_energy_breakdown_input_regs (pJ)':regs_input.w_cost/(activation_precision/input_precision)/1000, #pJ
            'peak_energy_breakdown_output_regs (pJ)':regs_output.w_cost/(activation_precision/input_precision)/1000, #pJ
            'peak_energy_breakdown_cache_2_input_regs (pJ)':regs_in_cache/(activation_precision/input_precision)/1000, #pJ
            'peak_energy_breakdown_output_regs_2_cache (pJ)':regs_out_cache/(activation_precision/input_precision)/1000, #pJ
            'peak_energy_breakdown_bank (pJ)':      dimc_macro.energy_bank/1000,
            'peak_energy_breakdown_mults (pJ)'  :   dimc_macro.energy_mults/1000,
            'peak_energy_breakdown_pv_adder_tree (pJ)': dimc_macro.energy_pv_adders/1000,
            'peak_energy_breakdown_adder_tree (pJ)':dimc_macro.energy_adder_tree/1000,
            'peak_energy_breakdown_accumulator (pJ)': dimc_macro.energy_accumulator/1000,
            'peak_energy_breakdown_regs_accumulator (pJ)': dimc_macro.energy_regs_accumulator/1000,
            'peak_energy_mac (fJ/MAC)':                     energy/number_of_ops*2,
            'peak_energy_breakdown_bank_mac (fJ/MAC)':      dimc_macro.energy_bank/number_of_ops*2,
            'peak_energy_breakdown_mults_mac (fJ/MAC)'  :   dimc_macro.energy_mults/number_of_ops*2,
            'peak_energy_breakdown_adder_tree_mac (fJ/MAC)':dimc_macro.energy_adder_tree/number_of_ops*2,
            'peak_energy_breakdown_accumulator_mac (fJ/MAC)': dimc_macro.energy_accumulator/number_of_ops*2,
            'peak_energy_breakdown_regs_accumulator_mac (fJ/MAC)': dimc_macro.energy_regs_accumulator/number_of_ops*2,
            'peak_energy_breakdown_regs (fJ/MAC)': (regs_input.w_cost+regs_output.w_cost)/(activation_precision/input_precision)/number_of_ops *2,
            'peak_energy_breakdown_cache (fJ/MAC)': (regs_in_cache+regs_out_cache)/(activation_precision/input_precision)/number_of_ops *2
            }

    #breakpoint()
    return total_area, tclk, tops, topsw, topsmm2, mem_density, peak_dict

def get_imc_config_w_area_budget(cacti_path, activation_precision, input_precision, weight_precision, output_precision, cells_per_mult, number_of_cores, area_target, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd):
    '''
    search for maximum allowed dimc setting for given area budget
    :param cacti_path:              path where to run cacti
    :param activation_precision:    activation bit precision (unit: bit)
    :param  input_precision:        how many bits per input channel (unit: bit)
    :param weight_precision:        weight bit precision (unit: bit)
    :param output_precision:        output bit precision (unit: bit) (affect the size of output registers)
    :param cells_per_mult:          how many rows share one multiplier
    :param number_of_cores:         how many cores is the area budget used for
    :param area_target:             the area budget for imc macros
    :output result_input_channel:   the found values for input_channel for each output_channel
    :output result_output_channel:  the found values for output_channel
    :param unit_area_28nm:          the area cost of single NAND2 gate for 28nm process
    :param unit_delay_28nm:         the delay cost of single NAND2 gate for 28nm process
    :param unit_cap_28nm:           the capancitance of single NAND2 gate for 28nm process
    '''

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')    
    result_input_channel = []
    result_output_channel = []

    output_channel = 2
    # status: set to 'stop' when the smallest imc macro is still bigger than area_budget.
    status = 'start' # flag of starting searching
    while True:
        # check the area of minimum imc size
        min_row = 64 * 8 / (weight_precision * output_channel) # minimum cacti mem_size = 64B. Here is checking how many rows for 64B
        rows = max(32, min_row, cells_per_mult)
        input_channel = rows/cells_per_mult
        min_area = get_imc_cost(cacti_path, activation_precision, input_precision, weight_precision, output_precision, input_channel, output_channel, rows, 1, 0,  unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd)[0]
        if min_area > area_target/number_of_cores or input_channel < 2: # if min_area is still bigger, stop searching
            status = 'stop'
            break

        '''
        Note: a constraint on input_channel: it must be 2**x (due to the structure of adder tree)
        '''
        left_x = math.log(input_channel, 2)
        right_x = left_x * 2

        '''
        find the maximum x meeting the area budget, given area = f(x) and f(x) is an increment function
        find a range [left_x, right_x] which contains required x
        '''
        while True:
            input_channel = 2**left_x
            rows = input_channel * cells_per_mult
            logging.debug(f'[check, input_channel: {input_channel}]')
            try:
                left_y = get_imc_cost(cacti_path, activation_precision, input_precision, weight_precision, output_precision, input_channel, output_channel, rows, 1, 0,  unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd)[0]
            except:
                status = 'stop' # if during iteration, left_x becomes so small that triggers CACTI error, then no result and stop
                break
        
            input_channel = 2**right_x
            rows = input_channel * cells_per_mult
            right_y = get_imc_cost(cacti_path, activation_precision, input_precision, weight_precision, output_precision, input_channel, output_channel, rows, 1, 0, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd)[0]
        
            logging.debug(f'[left_x: {left_x}, right_x: {right_x}, left_y: {left_y}, right_y: {right_y}, area_target/number_of_cores: {area_target/number_of_cores}]')
            if left_y > area_target/number_of_cores:
                if left_x == 1: # 2 input channels still has bigger area than area_target, then no solution and stop
                    status = 'stop'
                    break
                else:
                    left_x = left_x // 2
                    right_x = left_x
            elif right_y < area_target/number_of_cores:
                left_x = right_x
                right_x = right_x * 2
            else:
                break # jump out while loop when already found [left_x, right_x] containing solution

        # stop exploration when the area budget cannot be met even with the smallest input channel
        if status == 'stop':
            break

        # search within [left_x, right_x]
        while True:
            mid_x = (left_x + right_x + 1) // 2
            input_channel = 2 ** mid_x
            rows = input_channel * cells_per_mult
            mid_y = get_imc_cost(cacti_path, activation_precision, input_precision, weight_precision, output_precision, input_channel, output_channel, rows, 1, 0, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd)[0]
            if right_x - left_x > 1:
                if mid_y >= area_target/number_of_cores:
                    right_x = mid_x
                else:
                    left_x = mid_x
            else:
                if mid_y == area_target/number_of_cores:
                    result_input_channel.append(2**right_x)
                    result_output_channel.append(output_channel)
                    break
                else:
                    result_input_channel.append(2**left_x)
                    result_output_channel.append(output_channel)
                    break
        output_channel = 2**(int(math.log2(output_channel))+1) # let output channel be in the power of 2

    '''
    sometimes, multiple results will have same input_channel but different output_channel
    only the biggest output_channel should be saved. Dismiss others.
    '''
    available_input_channel = []
    available_output_channel = []
    for i in range(0, len(result_input_channel)-1):
        if result_input_channel[i] == result_input_channel[i+1]:
            pass
        else:
            available_input_channel.append(result_input_channel[i])
            available_output_channel.append(result_output_channel[i])
    try:
        available_input_channel.append(result_input_channel[-1])
        available_output_channel.append(result_output_channel[-1])
    except:
        pass # result_input_channel may be empty

    #print(result_input_channel, result_output_channel)
    #pdb.set_trace()
    return available_input_channel, available_output_channel

if __name__ == '__main__':
    # search for maximum allowed input_channel and output_channel for given area budget
    # setting
    activation_precision, input_precision, weight_precision, output_precision = 8,1,8,8
    cacti_path = './cacti-master'
    unit_area_28nm = 0.614 # um2
    unit_delay_28nm = 0.0478 # ns
    unit_cap_28nm = 0.7 # fF
    vdd = 0.9 # V
    core = 1
    import matplotlib.pyplot as plt
    from termcolor import cprint
    import numpy as np
    colors = ['orange', 'mediumseagreen', u'#27E1C1', u'#ff7f0e', u'#CCD5AE', 'plum', u'#BDE0FE', u'#CDB4DB', u'#FAD2E1'] # 1st color: for macro peak; 2nd: system peak; 3 onwards: workloads
    fig, axs = plt.subplots(nrows=2, ncols=3)
    for ii_a, input_precision in enumerate([1,2,4,8]):
        area_list = []
        cache_area_list = []
        macro_area_list = []
        ich_list = []
        topsw_list = []
        tops_list = []
        topsmm2_list = []
        tclk_list = []
        for input_channel in [256]:
            cells_per_mult = 16
            output_channel = input_channel/8/2
            rows = input_channel * cells_per_mult
            cache_bw = max(input_channel*activation_precision*core, output_channel*output_precision*core) # bw is the maximum one
            mem_size = 256*1024 #byte (fixed)
            if mem_size*8/cache_bw<32: # not meet CACTI requirement
                continue
            total_area, tclk, tops, topsw, topsmm2, memd, out_dict = get_imc_cost(cacti_path, activation_precision, input_precision, weight_precision, output_precision, input_channel, output_channel, rows, 1, 0, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd,cores=core,eval_level='system')
            print(total_area)
            breakpoint()
            ich_list.append(f'{input_channel}$\\times${input_channel}')
            area_list.append(total_area)
            tclk_list.append(tclk)
            tops_list.append(tops)
            topsw_list.append(topsw)
            topsmm2_list.append(topsmm2)
            cache_area_list.append(out_dict['area_cache (mm2)'])
            macro_area_list.append(out_dict['area_macros (mm2)']+out_dict['area_input_regs (mm2)']+out_dict['area_output_regs (mm2)'])
            
            #print('total area:', get_imc_cost(cacti_path, activation_precision, input_precision, weight_precision, output_precision, input_channel, output_channel, rows, 1, 0, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd,cores=core,eval_level='macro')[0])
        axs[0][0].plot(ich_list, topsw_list, label=f'B$_{{cycle}}$={input_precision}', marker='s', linestyle=':', markeredgewidth=1.0, markeredgecolor='black', color=colors[ii_a])
        axs[0][1].plot(ich_list, tops_list, label=f'B$_{{cycle}}$={input_precision}', marker='s', linestyle=':', markeredgewidth=1.0, markeredgecolor='black', color=colors[ii_a])
        axs[0][2].plot(ich_list, topsmm2_list, label=f'B$_{{cycle}}$={input_precision}', marker='s', linestyle=':', markeredgewidth=1.0, markeredgecolor='black', color=colors[ii_a])
        axs[1][0].plot(ich_list, tclk_list, label=f'B$_{{cycle}}$={input_precision}', marker='s', linestyle=':', markeredgewidth=1.0, markeredgecolor='black', color=colors[ii_a])
        axs[1][1].plot(ich_list, area_list, label=f'B$_{{cycle}}$={input_precision}', marker='s', linestyle=':', markeredgewidth=1.0, markeredgecolor='black', color=colors[ii_a])

        # plot bar for area
        pos_bias = [-3,-1,1,3] # bar position bias
        bar_width = 0.1
        labels = np.array(ich_list) # x label
        # Create positions for the bars on the x-axis
        x_pos = np.arange(len(labels))
        # Plot the bars
        base = [0 for x in range(0, len(labels))]
        if ii_a == 0:
            etype = f'Area$_{{cache}}$'
        else:
            etype = None
        values = np.array(cache_area_list)
        axs[1][2].bar(x_pos+bar_width/2*pos_bias[ii_a], values, bar_width, label=etype, bottom = base, color='orange', edgecolor='black')
        base += values
        if ii_a == 0:
            etype = f'Area$_{{macro}} (DIMC)$'
        else:
            etype = None
        values = np.array(macro_area_list)
        axs[1][2].bar(x_pos+bar_width/2*pos_bias[ii_a], values, bar_width, label=etype, bottom = base, color='mediumseagreen', edgecolor='black')

    axs[1][1].set_yscale('log') # area curve
    fig.text(0.5,0.01,'macro size (DIMC)', fontsize=15, ha='center')
    axs[0][0].set_ylabel(f'TOP/s/W', fontsize=15)
    axs[0][1].set_ylabel(f'TOP/s', fontsize=15)
    axs[0][2].set_ylabel(f'TOP/s/mm$^2$', fontsize=15)
    axs[1][0].set_ylabel(f'T$_{{clk}}$', fontsize=15)
    axs[1][1].set_ylabel(f'Area (mm$^2$)', fontsize=15)
    axs[1][2].set_ylabel(f'Area (bar-chart) (mm$^2$)', fontsize=15)
    axs[0][1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4, frameon=True, fontsize=12)
    axs[1][2].legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4, frameon=True, fontsize=10)
    axs[1][2].set_xticks(x_pos)
    for x in range(2):
        for y in range(3):
            axs[x][y].grid(which='both')
            axs[x][y].set_axisbelow(True)
            axs[x][y].set_xticklabels(labels, rotation=30)
    #plt.tight_layout()
    plt.show()




