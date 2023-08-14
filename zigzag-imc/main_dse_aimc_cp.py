from classes.stages import *
import time
from multiprocessing import Process
import itertools
import argparse
import get_aimc_cost
import get_imc_cost
import get_cacti_cost
import generate_zigzag_hardware_aimc
import generate_zigzag_hardware
import generate_default_mapping
import dimc_basics
from pathlib import Path
import os
import json
import hashlib
import math
import random



def hash_cfg(hd):
    hd_encoded = json.dumps(hd, sort_keys=True).encode()
    hashed_hd = int(hashlib.sha512(hd_encoded).hexdigest(), 16)
    return hex(hash(hashed_hd))

def runner(model, mapping, accelerator, accelerator_type, workload_name, hd_hash, hd):
    # Initialize the logger
    import logging as _logging
    _logging_level = _logging.INFO
    # _logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging.basicConfig(level=_logging_level,
                         format=_logging_format)

    # Initialize the MainStage which will start execution.
    # The first argument of this init is the list of stages that will be executed in sequence.
    # The second argument of this init are the arguments required for these different stages.
    mainstage = MainStage([  # Initializes the MainStage as entry point
        ONNXModelParserStage,  # Parses the ONNX Model into the workload
        AcceleratorParserStage,  # Parses the accelerator
        CompleteSaveBestStage,  # Saves all received CMEs information to a json
        WorkloadStage,  # Iterates through the different layers in the workload
        SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
        MinimalEnergyStage,  # Reduces all CMEs, returning minimal latency one
        #TemporalOrderingConversionStage,
        LomaStage,  # Generates multiple temporal mappings (TM)
        CostModelStage  # Evaluates generated SM and TM through cost model
    ],
        onnx_model=model,  # required by ONNXModelParserStage
        mapping_path=mapping,  # required by ONNXModelParserStage
        accelerator_path=accelerator,  # required by AcceleratorParserStage
        dump_filename_pattern=(hd, f"outputs/{accelerator_type}/{workload_name}/{hd_hash}.pkl"),  # output file save pattern
        loma_lpf_limit=6  # required by LomaStage
    )

    # Launch the MainStage
    mainstage.run()


def mp_dp(workload, accelerator_type, rc):
    area_target, mem_ratio, m_factor, number_of_cores, activation_precision, input_precision, rows, cols, weight_mem = rc
    weight_precision = activation_precision # unit: bit
    output_precision = activation_precision # unit: bit
    dram_bw = cols * number_of_cores # unit: bit
    workloads_model_path = {'ae': 'onnx_workload/mlperf_tiny/deepautoencoder/deepautoencoder_inferred_model.onnx',
                'ds-cnn': 'onnx_workload/mlperf_tiny/ds_cnn/ds_cnn_inferred_model.onnx',
                'mobilenet': 'onnx_workload/mlperf_tiny/mobilenet_v1/mobilenet_v1_inferred_model.onnx',
                'mobilenet-v2': 'onnx_workload/mobilenetv2.onnx',
                'resnet8': 'onnx_workload/mlperf_tiny/resnet8/resnet8_inferred_model.onnx'}
    ops_workloads = {'ae': 532512, 'ds-cnn': 5609536, 'mobilenet': 15907840, 'resnet8': 25302272}
    unit_area_28nm = 0.614 #um2
    unit_delay_28nm = 0.0478 #ns
    unit_cap_28nm = 0.7 #fJ
    vdd = 0.9 # V
    if accelerator_type == 'aimc_max':
        max_adc_resolution = 'dynamic' # ADC resolution [dynamic: input_pres+0.5*log2(input_channel); else: input_pres]
    else:
        max_adc_resolution = 'fixed' # upper limit fix at act_pres
    weight_sparsity, input_toggle_rate = 0, 1 # reflected in peak performance in csv file, not used in zigzag
    if accelerator_type in ['aimc_max', 'aimc_min']:
        map_filepath, hd_filepath, zigzag_output_path, cacti_path ='./inputs/tinyml/default_mapping', './inputs/tinyml/AIMC_accelerator_template',f'./outputs/{accelerator_type}/', './cacti-master/'
    else:
        map_filepath, hd_filepath, zigzag_output_path, cacti_path ='./inputs/tinyml/default_mapping', './inputs/tinyml/DIMC_accelerator_template',f'./outputs/{accelerator_type}/', './cacti-master/'
    onnx_name = {'ae': 'deepautoencoder', 'ds-cnn': 'ds-cnn', 'mobilenet': 'mobilenet_v1', 'resnet8': 'resnet8'} # prefix of onnx model
    hash_tuple = (activation_precision, input_precision, rows, cols, m_factor, input_toggle_rate, weight_sparsity, vdd, max_adc_resolution, number_of_cores, random.randbytes(1))
    #####################################


    area_target_imc = area_target*(100 - mem_ratio)/100 # area budget for imc, unit: mm2
    #if accelerator_type in ['aimc_max','aimc_min']:
    #    res_in_chl, res_out_chl = get_aimc_cost.get_aimc_config_w_area_budget(cacti_path, \
    #            activation_precision, input_precision, weight_precision, \
    #            output_precision, m_factor, number_of_cores, \
    #            area_target_imc, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd, max_adc_resolution)
    #else: 
    #    res_in_chl, res_out_chl = get_imc_cost.get_imc_config_w_area_budget(cacti_path, \
    #            activation_precision, input_precision, weight_precision, \
    #            output_precision, m_factor, number_of_cores, \
    #            area_target_imc, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd)
    #rows = 7*[2**x for x in range(5,12)]
    #cols = [2**x for x in range(5,12) for y in range(0,7)]
    #rows = [256]
    #cols = [256]
    res_in_chl = [rows//m_factor]
    res_out_chl = [cols//activation_precision]
    print(f'res_in_chl, res_out_chl in {[x for x in zip(res_in_chl, res_out_chl)]}, M {m_factor} number_of_cores {number_of_cores} MEM_RATIO {mem_ratio}')

    for r_in_chl, r_out_chl in zip(res_in_chl, res_out_chl):
        #if r_in_chl < 8 or r_out_chl < 8:
        #    continue # remove these points, causing empty spatial unrolling for zigzag
        if accelerator_type in ['aimc_max', 'aimc_min']:
            area, tclk, tops, topsw, topsmm2, mem_density, peak_dict = get_aimc_cost.get_aimc_cost(cacti_path, \
                activation_precision, input_precision, weight_precision, \
                output_precision, r_in_chl, r_out_chl, r_in_chl*m_factor, \
                input_toggle_rate, weight_sparsity, unit_area_28nm, \
                unit_delay_28nm, unit_cap_28nm, vdd, max_adc_resolution, ini_hash=number_of_cores)
        else:
            area, tclk, tops, topsw, topsmm2, mem_density, peak_dict = get_imc_cost.get_imc_cost(cacti_path, \
                    activation_precision, input_precision, weight_precision, output_precision,\
                    r_in_chl, r_out_chl, r_in_chl*m_factor, input_toggle_rate, \
                    weight_sparsity, unit_area_28nm, unit_delay_28nm, unit_cap_28nm, vdd, ini_hash=number_of_cores)
        print(f'IMC Area {area:.7f}, tclk {tclk:5.1f} topsw {topsw:5.0f}')  
        # find l2 cache configuration for given area
        cache_bw = max(r_in_chl*activation_precision*number_of_cores, r_out_chl*output_precision*number_of_cores) # bw is the maximum one

        area_target_cache = area_target - number_of_cores * area # area: imc real area
        cwd = os.getcwd()
        flag_cache_setting_not_found = False
        while True:
            try:
                #cache_size, ac_time, cache_area, cache_rc, cache_wc = get_cacti_cost.get_cacti_cost_w_area_budget(cacti_path = cacti_path, mem_type = 'sram', area_target=area_target_cache, bw=cache_bw,\
                #        hd_hash=str(hash((area_target_cache, cache_bw, random.randbytes(1))))) # unit: bit, ns, mm2, nJ, nJ
                mem_size = 256*1024 #byte (fixed)
                cache_size = mem_size * 8
                ac_time, cache_area, cache_rc, cache_wc = get_cacti_cost.get_cacti_cost(cacti_path, 'sram', mem_size, cache_bw, hd_hash=str(hash(hash_tuple)))
                print(f'[INFO] cache setting found, size: {cache_size}-bit, bw: {cache_bw}. Area target={area_target_cache}')

                # weight mem
                if weight_mem == 'dram': # off-chip dram for weight storage
                    dram_energy_rc_bit = 3.7 # pJ/bit
                    dram_energy_wc_bit = 3.7 # pJ/bit
                    dram_area = 0
                    dram_size = 100*1024*1024*8 # bit (100MB)
                elif weight_mem == 'sram': # on-chip sram for weight storage
                    weight_mem_size = 256 * 1024 # byte (fixed@256KB)
                    w_ac_time, w_mem_area, w_rc, w_wc = get_cacti_cost.get_cacti_cost(cacti_path, 'sram', weight_mem_size, dram_bw, hd_hash=str(hash(hash_tuple)))
                    dram_energy_rc_bit = 1000*w_rc/dram_bw # nJ -> pJ/bit
                    dram_energy_wc_bit = 1000*w_wc/dram_bw # nJ -> pJ/bit
                    dram_area = w_mem_area
                    dram_size = weight_mem_size * 8 # bit
                else:
                    raise Exception(f'mem level for weight can only be defined as dram or sram.')
                break
            except FileNotFoundError: # this means cacti didn't find any solution for current 'bw' setting
                print(f'current bw of cache is too high to meet the area budget {area_target_cache}mm2 and CACTI did not find any topology. bw will be halved from {cache_bw} to {cache_bw//2} and rerun CACTI.')
                os.chdir(cwd) # recover to original work folder (or else error pops from get_cacti_cost)
                if cache_bw <= 8: # bw=8 is minimal for cache
                    print(f'but bw smaller than 8 is not permitted. Abort on this case.')
                    flag_cache_setting_not_found = True
                    break
                cache_bw = cache_bw//2
                cache_bw = int( math.ceil(cache_bw/8)*8 ) # guarantee cache_bw is 8n, or cacti will pops error.
        if flag_cache_setting_not_found == True:
            continue
        cache_rc = 1000 * cache_rc # unit: nJ -> pJ
        cache_wc = 1000 * cache_wc # unit: nJ -> pJ

        #cache_rc = cache_rc / cache_bw
        #cache_wc = cache_wc / cache_bw

        '''
        generate aimc hardware definition for zigzag
        '''
        hd = {
                'accelerator_type': accelerator_type,
                'area' : area, 
                'tclk' : tclk, 
                'tops': tops,
                'topsw' : topsw, 
                'topsmm2': topsmm2, 
                'mem_density': mem_density, 
                'peak_dict': peak_dict,
                'area_target_imc': area_target_imc,
                'area_target_cache': area_target_cache,
                'area_target': area_target,
                'cache_bw': cache_bw,
                'workload': workload,
                'act_pres':         activation_precision,
                'weight_pres':      weight_precision,
                'vdd':              vdd,
                'cells_per_mult':   m_factor,
                'input_precision':  input_precision,
                'output_pres':      output_precision,
                'rows':             int(r_in_chl * m_factor),
                'cols':             int(r_out_chl * weight_precision),
                'r_in_chl': r_in_chl,
                'r_out_chl': r_out_chl,
                'macros':           number_of_cores,
                'dram_size':        dram_size, # bit
                'dram_energy_wc_bit':  dram_energy_wc_bit, # dram energy/bit for read
                'dram_energy_rc_bit':  dram_energy_rc_bit, # dram energy/bit for write
                'dram_bw':          dram_bw, # bit
                'dram_area':        dram_area, # mm2
                'l2_sram_size':     cache_size, # bit
                'l2_sram_bw':       cache_bw,
                'l2_sram_r_cost':   cache_rc, # pJ/access
                'l2_sram_w_cost':   cache_wc, # pJ/access
                'l2_sram_area':     cache_area, # mm2
                'input_reg_area':   activation_precision * dimc_basics.UnitDff(unit_area_28nm, unit_delay_28nm, unit_cap_28nm).calculate_area(),
                'input_reg_w_cost': dimc_basics.UnitDff(unit_area_28nm, unit_delay_28nm, unit_cap_28nm).calculate_cap()/1000 * vdd**2, # pJ/bit
                'output_reg_area':  output_precision * dimc_basics.UnitDff(unit_area_28nm, unit_delay_28nm, unit_cap_28nm).calculate_area(),
                'output_reg_w_cost':dimc_basics.UnitDff(unit_area_28nm, unit_delay_28nm, unit_cap_28nm).calculate_cap()/1000 * vdd**2 # pJ/bit
                }
        if accelerator_type in ['aimc_max', 'aimc_min']:
                hd['adc_resolution'] = peak_dict['adc_resolution']

        hash_val = hash_cfg(hd)
        if accelerator_type in ['aimc_max', 'aimc_min']:
            generate_zigzag_hardware_aimc.generate_zigzag_hardware_aimc(filepath = hd_filepath + str(hash_val) + '.py', hd=hd)
        else:
            generate_zigzag_hardware.generate_zigzag_hardware(filepath = hd_filepath + str(hash_val) + '.py', hd=hd)
        generate_default_mapping.generate_default_mapping(filepath = map_filepath + str(hash_val) + '.py', precision = hd['weight_pres'])

        if accelerator_type in ['aimc_max', 'aimc_min']:
            runner(workloads_model_path[workload], f'inputs.tinyml.default_mapping{str(hash_val)}', f'inputs.tinyml.AIMC_accelerator_template{str(hash_val)}', accelerator_type, workload, str(hash_val), hd)
        else:
            runner(workloads_model_path[workload], f'inputs.tinyml.default_mapping{str(hash_val)}', f'inputs.tinyml.DIMC_accelerator_template{str(hash_val)}', accelerator_type, workload, str(hash_val), hd)



def main():
    #for accelerator_type in ['dimc', 'aimc_max']:
    for accelerator_type in ['aimc_max']:
        #for workload in ['mobilenet', 'ae', 'ds-cnn', 'resnet8']:
        for workload in ['ds-cnn']:
            Path(f"outputs/{accelerator_type}/{workload}/").mkdir(parents=True,exist_ok=True)
            #area_target = [0.1, 0.3, .5, .7,.9]
            #mem_ratio = [10, 30, 50, 70, 90]
            area_target = [0.0000001]
            mem_ratio = [0.00001]
            m_factor = [1]
            number_of_cores = [1]
            act_pres = [8] # equal to weight_pres, out_pres
            input_pres = [2] # input precision per channel
            rows = [2**x for x in range(5,11)]
            cols = [2**x for x in range(5,11)]
            
            rc_list = list(itertools.product(*[area_target, mem_ratio, m_factor, number_of_cores, act_pres, input_pres, rows, cols]))

            ## MULTIPROCESSING SECTION
            chunks = 50
            rc_list_chunk = [rc_list[i:i + chunks] for i in range(0, len(rc_list), chunks)]

            for rcc in rc_list_chunk:
                TIMEOUT = 3600
                ############################
                # for debug
                #os.system(f'rm -rf outputs/{accelerator_type}/{workload}/')
                #os.makedirs(f'outputs/{accelerator_type}/{workload}/', exist_ok=True)
                #for rc in rcc:
                #    if rc[-1] != rc[-2]: # nb_rows != nb_cols
                #        continue
                #    print(rc)
                #    mp_dp(workload, accelerator_type, rc)
                #    #breakpoint()
                #raise Exception('manual stop')
                ############################
                start = time.time()
                procs = [Process(target=mp_dp, args=(workload, accelerator_type, rc))  for rc in rcc if rc[-1]==rc[-2]]

                for p in procs: p.start()
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


if __name__ == "__main__":
    main()

### log
# 04182023: for reason I dont know, some cases cannot generate IMC with setting bw=8, size=64B
