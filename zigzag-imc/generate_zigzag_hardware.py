


def generate_zigzag_hardware(filepath, hd):
    con = """
import os
from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from classes.hardware.architecture.memory_level import MemoryLevel
from classes.hardware.architecture.operational_unit import Multiplier, AIMC, DIMC
from classes.hardware.architecture.operational_array import MultiplierArray, AIMCArray, DIMCArray
from classes.hardware.architecture.memory_instance import MemoryInstance, MemoryInstanceClocked
from classes.hardware.architecture.accelerator import Accelerator
from classes.hardware.architecture.core import Core


def multiplier_array():
    '''
    DIMC array variables
    :param multiplier_input_precision: activation precision, weight precision
    :param DIMC_unit_costs['vdd']:      supply voltage
    :param DIMC_unit_costs['CORE_ROWS']:how many rows share one input channel
    :param DIMC_unit_costs['INPUT_BITS_PER_CYCLE']: bit precision per input channel, must be 1 (or the adder tree cost and adder tree input precision is not correct)
    :param DIMC_unit_costs['WEIGHT_BITCELL']: storage precision
    :param DIMC_area (no use): area/cell(no use), area/adder/output channel, area/1b multiplier
    :param dimensions:                  D1: cols (output_channel*WEIGHT_BITCELL), D2: rows (input_channel*CORE_ROWS), D3: #macros
    :param technology:                  28 (must be 28, or the w_cost extracted from CACTI is not correct)
    :param imc_type:                    IMC
    '''
    multiplier_input_precision = [{act_pres}, {weight_pres}]
    DIMC_unit_costs = {{'vdd' : {vdd},
            'CORE_ROWS' : {cells_per_mult},
            'INPUT_BITS_PER_CYCLE': {input_precision},
            'WEIGHT_BITCELL' : {weight_pres}
            }}

    DIMC_area = {{'cell': 1, 'adder': 1, 'multiplier': 1}}
    dimensions = {{'D1': {cols}, 'D2':{rows}, 'D3': {macros}}}
    technology = 28
    imc_type = 'IMC' # not used
    dimc = DIMC(multiplier_input_precision, DIMC_unit_costs, DIMC_area, technology)
    dimc_array = DIMCArray(dimc, dimensions)
    return dimc_array




def memory_hierarchy(multiplier_array):
    '''Memory hierarchy variables'''
    ''' size = #bit '''
    ''' cost unit: pJ '''
    ''' area unit: mm2 '''
    # dram cost is not from cacti (size: 100 MB) (cost unit: pJ)
    # Reason: considering different DRAM size will have different r_cost/w_cost from CACTI, even with the same IO bandwidth. Here we use a constant value for pJ/bit for simplicity.
    # dram size is set to big, so it is enough for storing all weight/input/output
    # 3.7 pJ/bit is from: https://my.eng.utah.edu/~cs7810/pres/14-7810-02.pdf (P8)
    dram = MemoryInstance(name="weight_mem", size={dram_size}, r_bw={dram_bw}, w_bw={dram_bw}, r_cost={dram_energy_rc_bit}*{dram_bw}, w_cost={dram_energy_wc_bit}*{dram_bw}, area={dram_area},
                          r_port=2, w_port=2, rw_port=0, latency=1, min_r_granularity={dram_bw}//16, min_w_granularity={dram_bw}//16)

    # SRAM
    L2_sram = MemoryInstance(name='L2', 
            size={l2_sram_size}, 
            r_bw={l2_sram_bw}, w_bw={l2_sram_bw}, 
            r_cost={l2_sram_r_cost}, w_cost={l2_sram_w_cost}, 
            r_port = 3, w_port = 3, 
            area={l2_sram_area}, min_r_granularity={l2_sram_bw}//16, min_w_granularity={l2_sram_bw}//16)  # let me assume there are 16 banks
    

    # Input buffer
    input_buffer = MemoryInstance(name='input_buffer', 
            size={act_pres}, 
            r_bw={act_pres}, w_bw={act_pres}, 
            r_cost=0, w_cost={input_reg_w_cost}*{act_pres}, 
            r_port = 1, w_port = 1, 
            area={input_reg_area}) 

    # Output buffer
    output_buffer = MemoryInstance(name='output_buffer', 
            size={output_pres}, 
            r_bw={output_pres}, w_bw={output_pres}, 
            r_cost=0, w_cost={output_reg_w_cost}*{output_pres}, 
            r_port = 2, w_port = 2, 
            area={output_reg_area}) 

    # Create memory hierarchy graph object
    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    # Add DIMC array weight_cell
#    memory_hierarchy_graph.add_memory(memory_instance=weight_cell, operands=('I1',), 
#            served_dimensions=set())
    # Add Input buffer
    memory_hierarchy_graph.add_memory(memory_instance=input_buffer, operands=('I2',), 
            port_alloc=( {{'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None}}, ),
            served_dimensions={{(1,0,0)}})
    # Add output buffer
    memory_hierarchy_graph.add_memory(memory_instance=output_buffer, operands=('O',), 
            port_alloc=({{'fh':'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'}},), 
            served_dimensions={{(0,1,0)}})
    # Add L2 memory
    memory_hierarchy_graph.add_memory(memory_instance=L2_sram, operands=('I2','O',), 
             port_alloc=({{'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None}},
                         {{'fh': 'w_port_2', 'tl': 'r_port_3', 'fl': 'w_port_3', 'th': 'r_port_2'}},),
            served_dimensions='all')
    # Add DRAM
    memory_hierarchy_graph.add_memory(memory_instance=dram, operands=('I1', 'I2', 'O',),
                                      port_alloc=({{'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None}},
                                                  {{'fh': 'w_port_2', 'tl': 'r_port_2', 'fl': None, 'th': None}},
                                                  {{'fh': 'w_port_2', 'tl': 'r_port_2', 'fl': 'w_port_2', 'th': 'r_port_2'}},),
                                      served_dimensions='all')

    from visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph
    # visualize_memory_hierarchy_graph(memory_hierarchy_graph)

    return memory_hierarchy_graph


def cores():
    multiplier_array1 = multiplier_array()
    memory_hierarchy1 = memory_hierarchy(multiplier_array1)
    core1 = Core(1, multiplier_array1, memory_hierarchy1)
    return {{core1}}


acc_name = 'MyAccelerator'
acc_cores = cores()
global_buffer = None
accelerator = Accelerator(acc_name, acc_cores, global_buffer)

a = 1
""".format(**hd)

    f= open(filepath, 'w+')
    f.write(con)
    f.close()
