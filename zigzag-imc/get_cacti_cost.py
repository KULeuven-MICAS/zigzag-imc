import os
from cacti_config_creator import CactiConfig
import math
import pdb

def get_cacti_cost(cacti_path, mem_type, mem_size_in_byte, bw, hd_hash="a"):
    '''
    extract time, area, r_energy, w_energy cost from cacti 7.0 for 28nm
    :param cacti_path:          the location of cacti
    :param mem_type:            memory type (sram or dram)
    :param mem_size_in_byte:    memory size (unit: byte)
    :param bw:                  memory IO bitwidth
    Attention: for CACTI, the miminum mem_size=64B, minimum_rows=32
    '''
    # get the current working directory
    cwd = os.getcwd()

    # change the working directory
    os.chdir(cacti_path)

    # input parameters definition
    tech = 0.032 # technology: 32 nm (corresponding VDD = 0.9)
    if mem_type == 'dram':
        mem = '"main memory"'
    elif mem_type == 'sram':
        mem = '"ram"'
    else:
        msg = f'mem_type can only be dram or sram. Now it is: {mem_type}'
        raise ValueError(msg)

    """
    due to the growth of the area cost estimation from CACTI exceeds 1x when bw > 32, it will be set to 1x.
    """
    # check if bw > 32
    if bw > 32: # adjust the setting for CACTI
        rows = mem_size_in_byte * 8/bw
        line_size = int(32/8)
        IO_bus_width = 32
        mem_size_in_byte_adjust = rows * 32 / 8
    else: # normal case
        rows = mem_size_in_byte * 8/bw
        line_size = int(bw/8) # how many bytes on a row
        IO_bus_width = bw
        mem_size_in_byte_adjust = mem_size_in_byte

    file_path = '../self_gen' # location for input file (cache.cfg) and output file (cache.cfg.out)
    #breakpoint()
    os.system(f'rm -f {file_path}/cache_{hd_hash}.cfg.out') # clear output file
    #os.system(f'\cp {file_path}/cache.cfg {file_path}/cache_{hd_hash}.cfg') # clear output file
    C = CactiConfig()
    C.cacti_auto(['single', [['technology', 'cache_size', 'line_size', 'IO_bus_width', 'mem_type'], [tech, mem_size_in_byte_adjust, line_size, IO_bus_width, mem]]], f"{file_path}/cache_{hd_hash}.cfg")
    # read out result
    try:
        f = open(f'{file_path}/cache_{hd_hash}.cfg.out', 'r')
    except:
        msg = f'cacti failed. [setting] rows: {rows}, bw: {bw}, mem size (byte): {mem_size_in_byte}'
        print(msg)
    result = {}
    raw_result = f.readlines()
    f.close()
    for ii, each_line in enumerate(raw_result):
        if ii == 0:
            attribute_list = each_line.split(',')
            for each_attribute in attribute_list:
                result[each_attribute] = []
        else:
            for jj, each_value in enumerate(each_line.split(',')):
                try:
                    result[attribute_list[jj]].append(float(each_value))
                except:
                    pass
    # get required cost (0.9*0.9 accounts for cost scaling from 32 nm to 28 nm)
    try:
        access_time = 0.9*0.9*float(result[' Access time (ns)'][0]) # unit: ns (never used)
        if bw > 32:
            area    = 0.9*0.9*float(result[' Area (mm2)'][0]) * 2 * bw/32 # unit: mm2
            r_cost  = 0.9*0.9*float(result[' Dynamic read energy (nJ)'][0]) * bw/32 # unit: nJ
            w_cost  = 0.9*0.9*float(result[' Dynamic write energy (nJ)'][0]) * bw/32 # unit: nJ
        else:
            area    = 0.9*0.9*float(result[' Area (mm2)'][0]) * 2 # unit: mm2
            r_cost  = 0.9*0.9*float(result[' Dynamic read energy (nJ)'][0]) # unit: nJ
            w_cost  = 0.9*0.9*float(result[' Dynamic write energy (nJ)'][0]) # unit: nJ
    except KeyError:
        print(f'**KeyError** in result, current result: {result}')
        breakpoint()

    # change back the working directory
    os.system(f'rm {file_path}/cache_{hd_hash}.cfg.out') # clear output file
    os.system(f'rm {file_path}/cache_{hd_hash}.cfg') # clear output file
    os.chdir(cwd)

    return access_time, area, r_cost, w_cost

def get_cacti_cost_w_area_budget(cacti_path, mem_type, area_target, bw, hd_hash="a"):
    '''
    search and return delay, area, r_cost, w_cost of certain memory size satifying a defined area budget
    extract time, area, r_energy, w_energy cost from cacti 7.0
    :param cacti_path:          the location of cacti
    :param mem_type:            memory type (sram or dram)
    :param area_target:         area budget for the memory (unit: mm2)
    :param bw:                  memory IO bitwidth
    Attention: for CACTI, the miminum mem_size=64B, minimum_rows=32
    '''

    # input parameters definition
    tech = 0.032 # technology: 32 nm (corresponding VDD = 0.9)
    if mem_type == 'dram':
        mem = '"main memory"'
    elif mem_type == 'sram':
        mem = '"ram"'
    else:
        raise ValueError(f'mem_type can only be dram or sram. Now it is: {mem_type}')

    row = 32

    status = 'start' # flag of starting searching
    left_x = row # initilization (32 is minimum for CACTI)
    right_x = left_x * 2
    while True:
        '''
        find the maximum #rows (x) meeting the area budget, given area = f(x) and f(x) is an increment function
        find a range [left_x, right_x] which contains required x
        '''
        mem_size_in_byte_left_x = left_x * bw / 8
        try:
            ac_time, left_y, rc, wc = get_cacti_cost(cacti_path, mem_type, mem_size_in_byte_left_x, bw, hd_hash=hd_hash)
        except:
            status = 'stop' # if during iteration, left_x becomes so small that triggers CACTI error, then no result and stop


        mem_size_in_byte_right_x = right_x * bw / 8
        ac_time, right_y, rc, wc = get_cacti_cost(cacti_path, mem_type, mem_size_in_byte_right_x, bw, hd_hash=hd_hash)

        if left_y > area_target:
            left_x = left_x // 2
            right_x = left_x
        elif right_y < area_target:
            left_x = right_x
            right_x = right_x * 2
        else:
            break # jump out while loop when already found [left_x, right_x] containing solution

    # search within [left_x, right_x]
    while True:
        mid_x = (left_x + right_x + 1) // 2
        mem_size_in_byte_mid_x = mid_x * bw / 8
        ac_time, mid_y, rc, wc = get_cacti_cost(cacti_path, mem_type, mem_size_in_byte_mid_x, bw,hd_hash=hd_hash)
        if right_x - left_x > 1:
            if mid_y >= area_target:
                right_x = mid_x
            else:
                left_x = mid_x
        else:
            if mid_y == area_target:
                result = right_x # maximum allowed #rows
                break
            else:
                result = left_x # maximum allowed #rows
                break


    mem_size_in_bit_result = result * bw
    mem_size_in_byte_result = mem_size_in_bit_result / 8
    access_time, result_y, r_cost, w_cost = get_cacti_cost(cacti_path, mem_type, mem_size_in_byte_result, bw,hd_hash=hd_hash)
    # print(result_y)

    return mem_size_in_bit_result, access_time, result_y, r_cost, w_cost

if __name__ == '__main__':
    # an example for use
    '''
    find choice if given area
    '''
    #print( get_cacti_cost_w_area_budget(cacti_path = '../cacti-master', mem_type = 'sram', area_target=0.05, bw=1024/2) )
    '''
    print out access_time (ns), area (mm2), r_cost (nJ), w_cost (nJ)
    '''
    #print(get_cacti_cost(cacti_path = '../cacti-master', mem_type = 'dram', mem_size_in_byte = 200*1024, bw = 128))
    #rows = 128 
    #bw = 16
    for bw in [32]:
        mem_size = 32*32/8 # byte
        rows = mem_size*8/bw
        access_time, area, r_cost, w_cost = get_cacti_cost(cacti_path = './cacti-master', mem_type = 'sram', mem_size_in_byte = mem_size, bw = bw)
        print(f'area (mm2): {area}, r_cost (pJ)/bit: {r_cost*1000/bw}, w_cost (pJ)/bit: {w_cost*1000/bw}')
        print(f'{r_cost*1000}, {w_cost*1000}')
        #print(f'rows: {rows}, bw: {bw}, r_cost (pJ)/bit: {r_cost*1000/bw}, w_cost (pJ)/bit: {w_cost*1000/bw}')
