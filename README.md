# ZigZag-IMC
This repository presents the modified version of our tried-and-tested HW Architecture-Mapping Design Space Exploration (DSE) Framework for In-Memory Computing (IMC). ZigZag-IMC bridges the gap between algorithmic DL decisions and their acceleration cost on specialized accelerators through a fast and accurate HW cost estimation. 

A crucial part in this is the mapping of the algorithmic computations onto the computational memory array and memories. In the framework, multiple engines are provided that can automatically find optimal mapping points in this search space.

In this novel version, we have: 
- Added a comprehensive and validated hardware cost model for analog and digital IMC.
- Overhauled our HW architecture definition to:
    - include 2D-dimensional IMC arrays.
- Enhanced the cost model to support complex memories with variable port structures.
- Revamped the whole project structure to be more modular.

For a clear view on the file directory, please read file "file_directory_description.txt".

## Environment

We recommend setting up an anaconda environment.

A `conda-spec-file-{platform}.txt` is provided to set up the environment for linux64 and win64.

`$ conda create --name myzigzagenv --file conda-spec-file-{platform}.txt`

Alternatively, you can also install all packages directly through pip using the pip-requirements.txt with the command:

`$ pip install -r requirements.txt`

## Script
File "plot_figure.py" contains scripts for generating diagrams in the paper. File "sim_auto.py" contains scripts for running simulation (function "workload_eval_multi_processing") and output extraction.

## Simulation

Use script "sim_auto.py" to run simulation automatically. After putting the onnx file under onnx_workload/ and defining the hardware parameters in "sim_auto.py", the script will automatically create input hardware template in folder inputs/ and load in the onnx file. 

The output file will be in .pkl format under folder outputs/. Function "layer_info_read" in file "plot_figure.py" is used to read a specific layer in the output file; Function "workload_output_read" in file "sim_auto.py" is used to read the output of the entire workload.

For the details of the hardware definition, there is a template for DIMC in file "generate_zigzag_hardware.py" and for AIMC in file "generate_zigzag_hardware_aimc.py".

