import onnx 
from onnx import shape_inference
model = onnx.load("mobilenet_v1.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "mobilenet_v1_inferred_model.onnx")
