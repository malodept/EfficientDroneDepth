# src/edd/eval.py
import argparse, time, torch
from rich import print
from .data import make_loaders
from .modeling import DPTSmall, silog_loss, l1_masked, depth_metrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--bench", action="store_true")
    return ap.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_loader = make_loaders(args.data_root, img_size=args.img_size, batch_size=args.batch_size)
    model = DPTSmall(pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False); model.eval()
    total=0.0; m={"AbsRel":0.0,"RMSE":0.0,"Delta<1.25":0.0}; n=0; lat=[]
    for batch in val_loader:
        img=batch["image"].to(device); depth=batch["depth"].to(device); mask=batch["mask"].to(device)
        if args.bench:
            if device=="cuda": torch.cuda.synchronize()
            t0=time.perf_counter(); pred=model(img)
            if device=="cuda": torch.cuda.synchronize()
            lat.append((time.perf_counter()-t0)/img.size(0))
        else:
            pred=model(img)
        loss=0.7*silog_loss(pred,depth,mask)+0.3*l1_masked(pred,depth,mask)
        met=depth_metrics(pred,depth,mask); 
        for k in m: m[k]+=met[k]
        total+=loss.item(); n+=1
    for k in m: m[k]/=max(1,n)
    print(f"[eval] val={total/max(1,len(val_loader)):.4f} AbsRel={m['AbsRel']:.4f} RMSE={m['RMSE']:.4f} d1.25={m['Delta<1.25']:.4f}")
    if args.bench and lat:
        ms=1000.0*sum(lat)/len(lat); print(f"[bench] avg per-image latency {ms:.2f} ms  ~ {1000.0/ms:.1f} FPS")

if __name__ == "__main__":
    main()
