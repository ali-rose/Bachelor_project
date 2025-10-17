import onnx
import tensorrt as trt
def print_model_io(model_path, model_name):
    model = onnx.load(model_path)
    graph = model.graph
    
    print(f"Model: {model_name}")
    print("Inputs:")
    for input_tensor in graph.input:
        input_name = input_tensor.name
        input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"  Name: {input_name}, Shape: {input_shape}")
    
    print("Outputs:")
    for output_tensor in graph.output:
        output_name = output_tensor.name
        output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"  Name: {output_name}, Shape: {output_shape}")

if __name__ == "__main__":
    encoder_model_path = "/root/autodl-tmp/onnx_model/encoder_model.onnx"
    decoder_model_path = "/root/autodl-tmp/onnx_model/decoder_model.onnx"
    
    print_model_io(encoder_model_path, "Encoder Model")
    print_model_io(decoder_model_path, "Decoder Model")

    print(trt.__version__)

