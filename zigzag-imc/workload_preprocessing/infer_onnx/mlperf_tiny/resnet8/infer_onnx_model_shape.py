import onnx 
from onnx import shape_inference
model = onnx.load("resnet8.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "resnet8_inferred_model.onnx")
