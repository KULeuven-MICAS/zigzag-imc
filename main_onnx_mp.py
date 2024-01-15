from zigzag.classes.stages import *
import os
import random
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.ImcArray import ImcArray
from zigzag.classes.hardware.architecture.get_cacti_cost import get_w_cost_per_weight_from_cacti
from zigzag.classes.hardware.architecture.get_cacti_cost import get_cacti_cost
import numpy as np
import argparse
import re
from multiprocessing import Process, Value, Manager, Lock
from multiprocessing.managers import BaseManager, NamespaceProxy
import time
import sys
from termcolor import cprint



def memory_hierarchy_dut(imc_array, weight_sram_size, act_sram_size, visualize=False):
    """ [OPTIONAL] Get w_cost of imc cell group from CACTI if required """
    cacti_path = "zigzag/classes/cacti/cacti_master"
    tech_param = imc_array.unit.logic_unit.tech_param
    hd_param = imc_array.unit.hd_param
    dimensions = imc_array.unit.dimensions
    output_precision = hd_param["input_precision"] + hd_param["weight_precision"]
    if hd_param["enable_cacti"]:
        # unit: pJ/weight writing
        w_cost_per_weight_writing = get_w_cost_per_weight_from_cacti(cacti_path, tech_param, hd_param, dimensions)
    else:
        w_cost_per_weight_writing = hd_param["w_cost_per_weight_writing"] # user-provided value (unit: pJ/weight)

    """Memory hierarchy variables"""
    """ size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) """
    cell_group = MemoryInstance(
        name="cell_group",
        size=hd_param["weight_precision"] * hd_param["group_depth"],
        r_bw=hd_param["weight_precision"],
        w_bw=hd_param["weight_precision"],
        r_cost=0,
        w_cost=w_cost_per_weight_writing, # unit: pJ/weight
        area=0, # this area is already included in imc_array
        r_port=0, # no standalone read port
        w_port=0, # no standalone write port
        rw_port=1, # 1 port for both reading and writing
        latency=0, # no extra clock cycle required
    )
    reg_I1 = MemoryInstance(
        name="rf_I1",
        size=hd_param["input_precision"],
        r_bw=hd_param["input_precision"],
        w_bw=hd_param["input_precision"],
        r_cost=0,
        w_cost=tech_param["dff_cap"] * (tech_param["vdd"] ** 2) * hd_param["input_precision"], # pJ/access
        area=tech_param["dff_area"] * hd_param["input_precision"], # mm^2
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
    )

    reg_O1 = MemoryInstance(
        name="rf_O1",
        size=output_precision,
        r_bw=output_precision,
        w_bw=output_precision,
        r_cost=0,
        w_cost=tech_param["dff_cap"] * (tech_param["vdd"] ** 2) * output_precision, # pJ/access
        area=tech_param["dff_area"] * output_precision, # mm^2
        r_port=2,
        w_port=2,
        rw_port=0,
        latency=1,
    )

    ##################################### on-chip memory hierarchy building blocks #####################################

    sram_size = act_sram_size # unit: byte
    sram_bw = 128#max(imc_array.unit.bl_dim_size * hd_param["input_precision"] * imc_array.unit.nb_of_banks,
                  #imc_array.unit.wl_dim_size * output_precision * imc_array.unit.nb_of_banks)
    ac_time, sram_area, sram_r_cost, sram_w_cost = get_cacti_cost(cacti_path, tech_param["tech_node"], "sram",
                                                                  sram_size, sram_bw,
                                                                  hd_hash=str(hash((sram_size, sram_bw, random.randbytes(8)))))
    weight_sram_bw = 128#imc_array.unit.wl_dim_size * hd_param["weight_precision"] * imc_array.unit.nb_of_banks
    weight_ac_time, weight_sram_area, weight_sram_r_cost, weight_sram_w_cost = get_cacti_cost(cacti_path, tech_param["tech_node"], "sram",
                                                                  weight_sram_size, weight_sram_bw,
                                                                  hd_hash=str(hash((weight_sram_size, weight_sram_bw, random.randbytes(8)))))
 
    sram_256KB_256_3r_3w = MemoryInstance(
        name="sram_256KB",
        size=sram_size * 8, # byte -> bit
        r_bw=sram_bw,
        w_bw=sram_bw,
        r_cost=sram_r_cost,
        w_cost=sram_w_cost,
        area=sram_area,
        r_port=3,
        w_port=3,
        rw_port=0,
        latency=1,
        min_r_granularity=sram_bw//16, # assume there are 16 sub-banks
        min_w_granularity=sram_bw//16, # assume there are 16 sub-banks
    )
    weight_sram = MemoryInstance(
        name="weight_sram",
        size=weight_sram_size * 8, # byte -> bit
        r_bw=weight_sram_bw,
        w_bw=weight_sram_bw,
        r_cost=weight_sram_r_cost,
        w_cost=weight_sram_w_cost,
        area=weight_sram_area,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=sram_bw//16, # assume there are 16 sub-banks
        min_w_granularity=sram_bw//16, # assume there are 16 sub-banks
    )


    #######################################################################################################################

    dram_size = 1*1024*1024*1024 # unit: byte
    dram_ac_cost_per_bit = 3.7# unit: pJ/bit
    dram_bw = 32#imc_array.unit.wl_dim_size * hd_param["weight_precision"] * imc_array.unit.nb_of_banks
    dram_100MB_32_3r_3w = MemoryInstance(
        name="dram_1GB",
        size=dram_size*8, # byte -> bit
        r_bw=dram_bw,
        w_bw=dram_bw,
        r_cost=dram_ac_cost_per_bit*dram_bw, # pJ/access
        w_cost=dram_ac_cost_per_bit*dram_bw, # pJ/access
        area=0,
        r_port=3,
        w_port=3,
        rw_port=0,
        latency=1,
        min_r_granularity=dram_bw // 16,  # assume there are 16 sub-banks
        min_w_granularity=dram_bw // 16,  # assume there are 16 sub-banks
    )
    dram_100MB_32_1r_1w = MemoryInstance(
        name="dram_W_1GB",
        size=dram_size*8, # byte -> bit
        r_bw=dram_bw,
        w_bw=dram_bw,
        r_cost=dram_ac_cost_per_bit*dram_bw, # pJ/access
        w_cost=dram_ac_cost_per_bit*dram_bw, # pJ/access
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=dram_bw // 16,  # assume there are 16 sub-banks
        min_w_granularity=dram_bw // 16,  # assume there are 16 sub-banks
    )


    memory_hierarchy_graph = MemoryHierarchy(operational_array=imc_array)

    """
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    memory_hierarchy_graph.add_memory(
        memory_instance=cell_group,
        operands=("I2",),
        port_alloc=({"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},),
        served_dimensions=set(),
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_I1,
        operands=("I1",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions={(1, 0, 0)},
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_O1,
        operands=("O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},),
        served_dimensions={(0, 1, 0)},
    )

    ##################################### on-chip highest memory hierarchy initialization #####################################

    memory_hierarchy_graph.add_memory(
        memory_instance=sram_256KB_256_3r_3w,
        operands=("I1","O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_2", "tl": "r_port_2", "fl": "w_port_3", "th": "r_port_3"},
        ),
        served_dimensions="all",
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=weight_sram,
        operands=("I2",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
        ),
        served_dimensions="all",
    )


    ####################################################################################################################

    memory_hierarchy_graph.add_memory(
        memory_instance=dram_100MB_32_3r_3w,
        operands=("I1", "O"),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_3", "th": "r_port_3"},
        ),
        served_dimensions="all",
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=dram_100MB_32_1r_1w,
        operands=("I2",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
        ),
        served_dimensions="all",
    )

    if visualize:
        from zigzag.visualization.graph.memory_hierarchy import (
            visualize_memory_hierarchy_graph,
        )

        visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph


def imc_array_dut(array_size, m):
    """Multiplier array variables"""
    tech_param = { # 28nm
        "tech_node": 0.028,             # unit: um
        "vdd":      0.9,                # unit: V
        "nd2_cap":  0.7/1e3,            # unit: pF
        "xor2_cap": 0.7*1.5/1e3,        # unit: pF
        "dff_cap":  0.7*3/1e3,          # unit: pF
        "nd2_area": 0.614/1e6,          # unit: mm^2
        "xor2_area":0.614*2.4/1e6,      # unit: mm^2
        "dff_area": 0.614*6/1e6,        # unit: mm^2
        "nd2_dly":  0.0478,             # unit: ns
        "xor2_dly": 0.0478*2.4,         # unit: ns
        # "dff_dly":  0.0478*3.4,         # unit: ns
    }
    hd_param = {
        "pe_type":              "in_sram_computing",     # for in-memory-computing. Digital core for different values.
        "imc_type":             "digital",  # "digital" or "analog"
        "input_precision":      8,          # activation precision expected in the hardware
        "weight_precision":     8,          # weight precision expected in the hardware
        "input_bit_per_cycle":  1,          # nb_bits of input/cycle/PE
        "group_depth":          m,          # group depth in each PE
        "wordline_dimension": "D1",         # hardware dimension where wordline is (corresponds to the served dimension of input regs)
        "bitline_dimension": "D2",          # hardware dimension where bitline is (corresponds to the served dimension of output regs)
        "enable_cacti":         True,       # use CACTI to estimated cell array area cost (cell array exclude build-in logic part)
        "adc_resolution": max(1, 0.5 * np.log2(array_size['D1']))
        # Energy of writing weight. Required when enable_cacti is False.
        # "w_cost_per_weight_writing": 0.08,  # [OPTIONAL] unit: pJ/weight.
    }

    dimensions = {
        "D1": array_size['D1'],    # wordline dimension
        "D2": array_size['D2'],   # bitline dimension
        "D3": array_size['D3'],    # nb_macros (nb_arrays)
    }  # e.g. {"D1": ("K", 4), "D2": ("C", 32),}

    imc_array = ImcArray(
        tech_param, hd_param, dimensions
    )

    return imc_array

def cores_dut(array_size, m, weight_sram_size, act_sram_size):
    imc_array1 = imc_array_dut(array_size, m)
    memory_hierarchy1 = memory_hierarchy_dut(imc_array1, weight_sram_size, act_sram_size)

    core1 = Core(1, imc_array1, memory_hierarchy1)

    return {core1}


def runner(d1, d2, d3, m, wl_name, hw_name):
    # Get the onnx model, the mapping and accelerator arguments
    experiment_id = f"{hw_name}-{wl_name}"
    pkl_name = f'{experiment_id}-saved_list_of_cmes'


    array_size_cp = {'D1':d1, 'D2':d2, 'D3': d3}
    array_size_cp['D1'] /= m
    if not all([x >= 1 for x in array_size_cp.values()]):
        exit()
    # RESNET8
    if wl_name == 'resnet8':
        weight_sram_size = 77360
        act_sram_size = 32768 

    # deepautoencoder
    if wl_name == 'deepautoencoder':
        weight_sram_size = 264192
        act_sram_size = 768 + 1024

    # ds-cnn 
    if wl_name == 'ds_cnn':
        weight_sram_size = 22016
        act_sram_size = 16000

    # mobilenet v1
    if wl_name == 'mobilenet_v1':
        weight_sram_size = 208112
        act_sram_size = 55296


    print('Array size', array_size_cp)
    print('M factor', m)
    cores = cores_dut(array_size_cp, m, weight_sram_size, act_sram_size)
    acc_name = "1"
    accelerator = Accelerator(acc_name, cores)
    # Initialize the logger
    import logging as _logging
    _logging_level = _logging.INFO
    _logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging.basicConfig(level=_logging_level,
                         format=_logging_format)

    # Initialize the MainStage which will start execution.
    # The first argument of this init is the list of stages that will be executed in sequence.
    # The second argument of this init are the arguments required for these different stages.
    mainstage = MainStage([  # Initializes the MainStage as entry point
        ONNXModelParserStage,  # Parses the ONNX Model into the workload
        AcceleratorParserStage,  # Parses the accelerator
        PickleSaveStage, # Save CMEs to a pickle file
        # CompleteSaveStage,  # Saves all received CMEs information to a json
#                SearchUnusedMemoryStage, # Detect unnecessary memory instances
        WorkloadStage,  # Iterates through the different layers in the workload
#                RemoveUnusedMemoryStage, # Remove unnecessary memory instances
        MinimalEnergyStage, # Reduces all CMEs, returning minimal EDP one
        SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
        LomaStage,  # Generates multiple temporal mappings (TM)
        CostModelStage  # Evaluates generated SM and TM through cost model
    ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        workload=args.model,  # required by ONNXModelParserStage
        mapping=args.mapping,  # required by ONNXModelParserStage
        dump_filename_pattern=f"outputs/{experiment_id}-layer_?.json",  # output file save pattern
        pickle_filename=f"outputs_{wl_name}/dimc/{experiment_id}-layer_d1_{d1}_d2_{d2}_d3_{d3}_m_{m}.pkl",  # output file save pattern
        loma_lpf_limit=6,  # required by LomaStage
        loma_show_progress_bar=True,  # shows a progress bar while iterating over temporal mappings
        enable_mix_spatial_mapping_generation=True,  # True: enable generating mix spatial mapping. False: single layer dim mapping during the autogeneration
        maximize_hardware_utilization=False,  # True: only keep 2 sm with the highest hardware utilization to speedup simulation time
        enable_weight_diagonal_mapping=True,  # True: enable OX/OY unrolling when automatically generating sm
    )

    # Launch the MainStage
    mainstage.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup zigzag inputs")
    parser.add_argument('--model', metavar='path', required=True, help='path to onnx model, e.g. inputs/examples/my_onnx_model.onnx')
    parser.add_argument('--mapping', metavar='path', required=True, help='path to mapping file, e.g., inputs.examples.my_mapping')
    parser.add_argument('--accelerator', metavar='path', required=True, help='module path to the accelerator, e.g. inputs.examples.accelerator1')
    args = parser.parse_args()
    hw_name = args.accelerator.split(".")[-1]
    wl_name = re.split(r"/|\.", args.model)[-1]
    if wl_name == 'onnx':
        wl_name = re.split(r"/|\.", args.model)[-2]
    
    # ResNet8
    if wl_name == 'resnet8':
        array_sizes = [8*9, 16*9, 32*9, 64*9, 128*9]
        array_size_d3 = [1, 4, 8, 16]
        m_list = [1, 2, 4, 8, 16, 32]

    # AE
    if wl_name == 'deepautoencoder':
        array_sizes = [8, 32, 64, 128, 640]
        array_size_d3 = [1, 4, 8, 16]
        m_list = [1, 2, 4, 8, 16, 32]

    # DS CNN 
    if wl_name == 'ds_cnn':
        array_sizes = [9, 8*9, 16*9, 32*9, 64*9, 128*9]
        array_size_d3 = [1, 4, 8, 16]
        m_list = [1, 2, 4, 8, 16, 32]

    # MOBILENETV1
    if wl_name == 'mobilenet_v1':
        array_sizes = [9, 16*9, 32*9, 64*9, 128*9, 256*9]
        array_size_d3 = [1, 4, 8, 16]
        m_list = [1, 2, 4, 8, 16, 32]

    array_sizes = [9, 32, 8*9, 16*9,32*9,64*9]
    array_sizes = [32,8*9,16*9, 32*9,64*9]
    array_size_d3 = [1,4, 8,16,64]
    array_size_d3 = [4,16,8]
    m_list = [1,2,4,16,64]


    from itertools import product
   
    rc_list = list(product(*[array_sizes, array_sizes, array_size_d3, m_list]))
    chunks = 150
    rc_list_chunk = [rc_list[i:i + chunks] for i in range(0, len(rc_list), chunks)]

    for rcc in rc_list_chunk:
        TIMEOUT = 600
        start = time.time()
            
        procs = [Process(target=runner, args=(d1,d2,d3,m,wl_name,hw_name)) for d1,d2,d3,m in rcc]

        for p in procs : p.start()
        while time.time() - start <= TIMEOUT:
            if not any(p.is_alive() for p in procs):
                break
            time.sleep(1)
        else:
            print('TIMED OUT - KILLING ALL PROCESSES')
            for p in procs:
                p.terminate()
                p.join()
        for p in procs : p.join()

 