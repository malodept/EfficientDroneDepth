# src/edd/export_onnx.py
import argparse, torch, onnx
from .modeling import DPTSmall

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=384)
    return ap.parse_args()

def main():
    args = parse_args(); device="cpu"
    model = DPTSmall(pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False); model.eval()
    dummy = torch.randn(1,3,args.img_size,args.img_size, device=device)
    torch.onnx.export(model, dummy, args.onnx, input_names=["input"], output_names=["depth"],
                      dynamic_axes={"input":{0:"batch"}, "depth":{0:"batch"}}, opset_version=17)
    onnx_model = onnx.load(args.onnx); onnx.checker.check_model(onnx_model)
    print(f"[onnx] {args.onnx}")

if __name__ == "__main__":
    main()
