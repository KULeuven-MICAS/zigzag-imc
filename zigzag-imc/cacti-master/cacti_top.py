from cacti_config_creator import CactiConfig
import os
import csv
import pdb
try:
    import pickle
except:
    os.system('pip install pickle')
    import pickle
try:
    import yaml
except:
    os.system('pip install pyyaml')
    import yaml


os.system('pwd')

file_path = './self_gen'
if not os.path.isdir('%s' % file_path):
    os.system('mkdir %s' % file_path)

os.system('rm -rf %s/*' % file_path)
C = CactiConfig()

'''Function 1: set default value'''
# C.change_default_value(['technology'], [0.090])

'''Function 2: use default values to run CACTI'''
# C.cacti_auto(['default'], file_path + '/cache.cfg')

'''Function 3: use user-defined + default values to run CACTI'''
# C.cacti_auto(['single', [['technology', 'cache_size'],[0.022, 524288]]], file_path+'/cache.cfg')

'''Function 4: sweep any one variable using the default list & other default value'''
# C.cacti_auto(['sweep', ['IO_bus_width']], file_path+'/cache.cfg')

''' Combining Function 1 & 4 to do multi-variable sweep '''
for tech in [0.032]:
    for line_size, IO_bus_width in [[2**yy, 2**yy*8] for yy in range(4,20)]: # yy: how many bytes per row, yy*8: IO bitwidth (how many bits per row)
        for rows in range(1024, 8192, 1024):
            mem_size = rows * line_size # memory size in Byte
            if mem_size < 64: # CACTI: mem_size must be >= 64 B for cache type "ram"
                continue
            C.cacti_auto(['single', [['technology', 'cache_size', 'line_size', 'IO_bus_width', 'mem_type'], [tech, mem_size, line_size, IO_bus_width, '"main memory"']]], file_path + '/cache.cfg') # regarding mem_type, "main memory" is for DRAM, "ram" is for SRAM.

''' read out result '''
f = open('%s/cache.cfg.out' % file_path, 'r')
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
print(result)

''' clear files '''
try:
    os.system('rm ./example_mem_pool.yaml')
except:
    pass
try:
    os.system('rm ./example_mem_pool.csv')
except:
    pass

for i in range(len(result[' Capacity (bytes)'])):
    mem_name = 'sram_BW%s_' %int(result[' Output width (bits)'][i]) + str(int(result[' Capacity (bytes)'][i])) + '_Byte'
    tech = result['Tech node (nm)'][i]
    size_bit = result[' Capacity (bytes)'][i]
    access_time = float(result[' Access time (ns)'][i])
    area = result[' Area (mm2)'][i]
    read_word = result[' Dynamic read energy (nJ)'][i]
    write_word = result[' Dynamic write energy (nJ)'][i]
    leak = result[' Standby leakage per bank(mW)'][i]
    mem_bw = result[' Output width (bits)'][i]
    mem_type = 'single_port_double_buffered'
    utilization_rate = 0.7
    new_result = {'%s' % mem_name: {
        'tech (nm)': int(tech),
        'size_bit': int(size_bit * 8),
        'access_time (ns)': access_time,
        'area (mm2)': area*2,
        'cost (nJ)': {'read_word': read_word, 'write_word': write_word},
        'leak (mW)': leak,
        'mem_bw (bits)': int(mem_bw),
        'mem_type': mem_type,
        'utilization_rate': utilization_rate,
        'nbanks': 1
    }}
    
    ''' result written to a yaml target '''
    with open('./example_mem_pool.yaml', 'a+') as file:
        yaml.dump(new_result, file)
        file.write('\n')

    ''' result written to a csv target '''
    ''' if no result for a specific configuration in output file, it means no valid data array organizations found '''
    with open('./example_mem_pool.csv', 'a+') as csv_file:
        fieldnames = ['Tech (nm)', 'Capacity (B)', 'Rows', 'Output width (bits)', 'Access time (ns)', 'Area (mm2)', 'Read energy (nJ)', 'Write energy (nJ)', 'Leakage power (mW)']
        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)

        ''' generate header '''
        if i == 0:
            writer.writeheader()
        ''' data output '''
        writer.writerow({'Tech (nm)': int(tech), 'Capacity (B)': int(size_bit), 'Rows': int(size_bit*8/mem_bw), 'Output width (bits)': int(mem_bw), 'Access time (ns)': access_time, 'Area (mm2)': area*2, 'Read energy (nJ)': read_word, 'Write energy (nJ)': write_word, 'Leakage power (mW)': leak})

''' result save to pickle file '''
# try:
#     import pandas as pd
# except:
#     os.system('pip install pandas')
#     import pandas as pd
# data = pd.read_csv('./example_mem_pool.csv')
with open('./example_mem_pool.pickle', 'wb') as pickle_file:
    pickle.dump(result, pickle_file)

