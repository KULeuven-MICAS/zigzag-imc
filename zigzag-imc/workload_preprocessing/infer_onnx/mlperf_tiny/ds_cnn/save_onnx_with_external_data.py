import onnx
model = onnx.load('../../tensorflow_to_onnx/ds_cnn/kws_ref_model_float32.onnx')

onnx.save_model(model, 'ds_cnn.onnx', save_as_external_data = True, all_tensors_to_one_file = True, location = 'dss_cnn_external_data.onnx', size_threshold = 1024, convert_attribute = False)
