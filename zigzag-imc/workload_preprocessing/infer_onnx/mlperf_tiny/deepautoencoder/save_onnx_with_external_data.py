import onnx
model = onnx.load('../../tensorflow_to_onnx/deepautoencoder/ad01_fp32.onnx')

onnx.save_model(model, 'deepautoencoder.onnx', save_as_external_data = True, all_tensors_to_one_file = True, location = 'deepautoencoder_external_data.onnx', size_threshold = 1024, convert_attribute = False)
