import onnx
model = onnx.load('../../tensorflow_to_onnx/mobilenet_v1/vww_96_float.onnx')

onnx.save_model(model, 'mobilenet_v1.onnx', save_as_external_data = True, all_tensors_to_one_file = True, location = 'mobilenet_v1_external_data.onnx', size_threshold = 1024, convert_attribute = False)
