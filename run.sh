#
# This file provides an overview what command is supported by current framework.
# For now, three kinds of hardware is supported.

# (1) pure digital hardware (outputs include energy estimation, number of cycles)
# Command example:
#python main_onnx.py --model zigzag/inputs/examples/workload/alexnet.onnx --accelerator zigzag.inputs.examples.hardware.TPU_like --mapping zigzag.inputs.examples.mapping.default

# (2) Digital in-sram computing hardware (outputs include energy estimation, number of cycles, clock time, area)
python main_onnx_opt.py --model zigzag/inputs/examples/workload/mlperf_tiny/resnet8.onnx --accelerator zigzag.inputs.examples.hardware.Dimc --mapping zigzag.inputs.examples.mapping.default_imc

# (3) Analog in-sram computing hardware (outputs include energy estimation, number of cycles, clock time, area)
#python main_onnx.py --model zigzag/inputs/examples/workload/mlperf_tiny/ds_cnn.onnx --accelerator zigzag.inputs.examples.hardware.Aimc --mapping zigzag.inputs.examples.mapping.default_imc

