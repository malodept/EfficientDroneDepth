# src/edd/quantize_dynamic.py
import argparse, onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    quantize_dynamic(model_input=args.onnx, model_output=args.out, weight_type=QuantType.QInt8)
    m = onnx.load(args.out); onnx.checker.check_model(m)
    print(f"[quant] wrote {args.out}")

if __name__ == "__main__":
    main()
