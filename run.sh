#
# This file provides an overview what command is supported by current framework.
# For now, three kinds of hardware is supported.

# (1) pure digital hardware
# Command example:
#python main.py --model zigzag.inputs.examples.workload.resnet18 --mapping zigzag.inputs.examples.mapping.default --accelerator zigzag.inputs.examples.hardware.TPU_like
python main_onnx.py --model zigzag/inputs/examples/workload/alexnet.onnx --accelerator zigzag.inputs.examples.hardware.TPU_like --mapping zigzag.inputs.examples.mapping.default

# (2) Digital in-sram computing hardware
python main_onnx.py --model zigzag/inputs/examples/workload/mlperf_tiny/ds_cnn.onnx --accelerator zigzag.inputs.examples.hardware.Dimc --mapping zigzag.inputs.examples.mapping.default_imc

# (3) Analog in-sram computing hardware
python main_onnx.py --model zigzag/inputs/examples/workload/mlperf_tiny/ds_cnn.onnx --accelerator zigzag.inputs.examples.hardware.Aimc --mapping zigzag.inputs.examples.mapping.default_imc

