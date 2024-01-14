#python main_onnx_mp.py --model zigzag/inputs/examples/workload/mlperf_tiny/deepautoencoder.onnx --mapping zigzag.inputs.examples.mapping.default_imc --accelerator zigzag.inputs.examples.hardware.Dimc
#python main_onnx_mp.py --model zigzag/inputs/examples/workload/mlperf_tiny/ds_cnn.onnx --mapping zigzag.inputs.examples.mapping.default_imc --accelerator zigzag.inputs.examples.hardware.Dimc
#python main_onnx_mp.py --model zigzag/inputs/examples/workload/mlperf_tiny/mobilenet_v1.onnx --mapping zigzag.inputs.examples.mapping.default_imc --accelerator zigzag.inputs.examples.hardware.Dimc
python main_onnx_mp.py --model zigzag/inputs/examples/workload/mlperf_tiny/resnet8.onnx --mapping zigzag.inputs.examples.mapping.default_imc --accelerator zigzag.inputs.examples.hardware.Dimc
