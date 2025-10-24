# src/edd/train.py
import os, time, argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich import print
from .data import make_loaders
from .modeling import DPTSmall, silog_loss, l1_masked, depth_metrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--limit_samples", type=int, default=None)
    ap.add_argument("--backbone", type=str, default="resnet34")
    ap.add_argument("--ckpt", type=str, default="runs/edd_midas.pt")
    ap.add_argument("--num_workers", type=int, default=0)
    return ap.parse_args()

def train_one_epoch(model, loader, optimizer, device):
    model.train(); total = 0.0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        pred = model(img)
        loss = 0.7 * silog_loss(pred, depth, mask) + 0.3 * l1_masked(pred, depth, mask)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, device):
    model.eval(); total = 0.0
    m = {"AbsRel":0.0,"RMSE":0.0,"Delta<1.25":0.0}; n=0
    for batch in loader:
        img = batch["image"].to(device); depth = batch["depth"].to(device); mask = batch["mask"].to(device)
        pred = model(img)
        loss = 0.7 * silog_loss(pred, depth, mask) + 0.3 * l1_masked(pred, depth, mask)
        metrics = depth_metrics(pred, depth, mask)
        for k in m: m[k] += metrics[k]
        total += loss.item(); n+=1
    for k in m: m[k] /= max(1,n)
    return total / max(1,len(loader)), m

def main():
    args = parse_args(); os.makedirs("runs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"[device] {device}")
    train_loader, val_loader = make_loaders(args.data_root, img_size=args.img_size, batch_size=args.batch_size, limit_samples=args.limit_samples)
    model = DPTSmall(backbone_name=args.backbone, pretrained=True).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=args.epochs)
    best = 9e9
    for epoch in range(1, args.epochs+1):
        t0=time.time(); tr = train_one_epoch(model, train_loader, optim, device)
        va, met = validate(model, val_loader, device); sched.step()
        print(f"[epoch {epoch}] train={tr:.4f} val={va:.4f} AbsRel={met['AbsRel']:.4f} RMSE={met['RMSE']:.4f} d1.25={met['Delta<1.25']:.4f} time={time.time()-t0:.1f}s")
        if va < best:
            best = va; torch.save({"model": model.state_dict(), "args": vars(args)}, args.ckpt); print(f"[ckpt] saved -> {args.ckpt}")
    print("[done]")

if __name__ == "__main__":
    main()
