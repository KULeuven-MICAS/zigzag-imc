import onnx 
from onnx import shape_inference
model = onnx.load("ds_cnn.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "ds_cnn_inferred_model.onnx")
