from zigzag.classes.stages import *
import argparse
import re

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

# Initialize the MainStage which will start execution.
# The first argument of this init is the list of stages that will be executed in sequence.
# The second argument of this init are the arguments required for these different stages.
mainstage = MainStage([  # Initializes the MainStage as entry point
    ONNXModelParserStage,  # Parses the ONNX Model into the workload
    AcceleratorParserStage,  # Parses the accelerator
    # PickleSaveStage, # Save CMEs to a pickle file
    CompleteSaveStage,  # Saves all received CMEs information to a json
    SearchUnusedMemoryStage, # Detect unnecessary memory instances
    WorkloadStage,  # Iterates through the different layers in the workload
    RemoveUnusedMemoryStage, # Remove unnecessary memory instances
    MinimalEDPStage, # Reduces all CMEs, returning minimal EDP one
    SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
    LomaStage,  # Generates multiple temporal mappings (TM)
    CostModelStage  # Evaluates generated SM and TM through cost model
],
    accelerator=args.accelerator,  # required by AcceleratorParserStage
    workload=args.model,  # required by ONNXModelParserStage
    mapping=args.mapping,  # required by ONNXModelParserStage
    dump_filename_pattern=f"outputs/{experiment_id}-layer_?.json",  # output file save pattern
    pickle_filename=f"outputs/{experiment_id}-layer_?.pkl",  # output file save pattern
    loma_lpf_limit=6,  # required by LomaStage
    loma_show_progress_bar=True,  # shows a progress bar while iterating over temporal mappings
    enable_mix_spatial_mapping_generation=True,  # True: enable generating mix spatial mapping. False: single layer dim mapping during the autogeneration
    maximize_hardware_utilization=False,  # True: only keep 2 sm with the highest hardware utilization to speedup simulation time
    enable_weight_diagonal_mapping=True,  # True: enable OX/OY unrolling when automatically generating sm
)

# Launch the MainStage
mainstage.run()
