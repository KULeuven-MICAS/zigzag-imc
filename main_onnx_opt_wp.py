from zigzag.classes.stages import *
from zigzag.classes.opt.NDO.black_box_optimizer import tID, OptimizerTarget
import os
import random
import argparse
import re
import numpy as np

d1_target = OptimizerTarget(target_stage = 'AcceleratorParserStage',
                target_object = [tID('object','accelerator'), tID('obj','cores'), tID('list',0), tID('obj','operational_array'), tID('obj','dimensions'), tID('list',0)],
                target_modifier = 'set_array_dim', target_range = np.array([1]))
d2_target = OptimizerTarget(target_stage = 'AcceleratorParserStage',
                target_object = [tID('object','accelerator'), tID('obj','cores'), tID('list',0), tID('obj','operational_array'), tID('obj','dimensions'), tID('list',1)],
                target_modifier = 'set_array_dim', target_range = np.array([24]))
d3_target = OptimizerTarget(target_stage = 'AcceleratorParserStage',
                target_object = [tID('object','accelerator'), tID('obj','cores'), tID('list',0), tID('obj','operational_array'), tID('obj','dimensions'), tID('list',2)],
                target_modifier = 'set_array_dim', target_range = np.array([384]))
g_target = OptimizerTarget(target_stage = 'AcceleratorParserStage',
                target_object = [tID('object','accelerator'), tID('obj','cores'), tID('list',0), tID('obj','operational_array')],
                target_modifier = 'set_group_depth', target_range = np.array([256]))



optimizer_targets = [d1_target, d2_target,d3_target,g_target]

# RESNET8
# Get the onnx model, the mapping and accelerator arguments
parser = argparse.ArgumentParser(description="Setup zigzag inputs")
parser.add_argument('--model', metavar='path', required=True, help='path to onnx model, e.g. inputs/examples/my_onnx_model.onnx')
parser.add_argument('--mapping', metavar='path', required=True, help='path to mapping file, e.g., inputs.examples.my_mapping')
parser.add_argument('--accelerator', metavar='path', required=True, help='module path to the accelerator, e.g. inputs.examples.accelerator1')
args = parser.parse_args()

# Initialize the logger
import logging as _logging
_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level,
                     format=_logging_format)

hw_name = args.accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", args.model)[-1]
if wl_name == 'onnx':
    wl_name = re.split(r"/|\.", args.model)[-2]
experiment_id = f"{hw_name}-{wl_name}"
pkl_name = f'{experiment_id}-saved_list_of_cmes'

optimizer_params = {'iterations':100,
    'init_temperature':1.,
    'end_temperature':0.1,
    'cooling_factor':0.95,
    'area_budget':1.,
    'optimizer_type':'simulated_annealing'}
optimizer_params = {'area_budget':1.,
    'optimizer_type':'grid_search'}
optimizer_params = {'iterations':600,
    'init_iterations':40,
    'area_budget':1.,
    'optimizer_type':'bayesian_optimization'}

#optimizer_params = {'iterations':10,
#    'area_budget':1.,
#    'optimizer_type':'particle_swarm_optimization'}


optimizer_params['optimizer_targets'] = optimizer_targets

# Initialize the MainStage which will start execution.
# The first argument of this init is the list of stages that will be executed in sequence.
# The second argument of this init are the arguments required for these different stages.
mainstage = MainStage([  # Initializes the MainStage as entry point
    #GDStage,
    #NDOStage,
    BBOStage,
    MultiProcessingGatherStage,
    ONNXModelParserStage,  # Parses the ONNX Model into the workload
    AcceleratorParserStage,  # Parses the accelerator
    WeightPackingStage,
    PickleSaveStage, # Save CMEs to a pickle file
    WorkloadStage,  # Iterates through the different layers in the workload
    MultiProcessingSpawnStage,
    MinimalEnergyStage, # Reduces all CMEs, returning minimal EDP one
    SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
    LomaStage,  # Generates multiple temporal mappings (TM)
    CostModelStage  # Evaluates generated SM and TM through cost model
],
    optimizer_params=optimizer_params,
    accelerator=args.accelerator,  # required by AcceleratorParserStage
    workload=args.model,  # required by ONNXModelParserStage
    mapping=args.mapping,  # required by ONNXModelParserStage
    dump_filename_pattern=f"outputs/{experiment_id}-layer_?.json",  # output file save pattern
    pickle_filename=f"outputs/{experiment_id}.pkl",  # output file save pattern
    loma_lpf_limit=6,  # required by LomaStage
    loma_show_progress_bar=True,  # shows a progress bar while iterating over temporal mappings
    enable_mix_spatial_mapping_generation=True,  # True: enable generating mix spatial mapping. False: single layer dim mapping during the autogeneration
    maximize_hardware_utilization=True,  # True: only keep 2 sm with the highest hardware utilization to speedup simulation time
    enable_weight_diagonal_mapping=False,  # True: enable OX/OY unrolling when automatically generating sm
)

# Launch the MainStage
mainstage.run()
