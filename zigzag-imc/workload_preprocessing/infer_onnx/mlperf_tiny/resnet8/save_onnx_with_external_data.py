import onnx
model = onnx.load('../../tensorflow_to_onnx/resnet8/pretrainedResnet.onnx')

onnx.save_model(model, 'resnet8.onnx', save_as_external_data = True, all_tensors_to_one_file = True, location = 'resnet8_external_data.onnx', size_threshold = 1024, convert_attribute = False)
