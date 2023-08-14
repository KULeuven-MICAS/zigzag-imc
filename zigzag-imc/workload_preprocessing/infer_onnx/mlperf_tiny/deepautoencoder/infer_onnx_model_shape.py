import onnx 
from onnx import shape_inference
model = onnx.load("deepautoencoder.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "deepautoencoder_inferred_model.onnx")
