# src/edd/train.py
import os, time, argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR
from rich import print
from .data import make_loaders
from .modeling import DPTSmall, silog_loss, l1_masked, depth_metrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--limit_samples", type=int, default=None)
    ap.add_argument("--backbone", type=str, default="resnet34")
    ap.add_argument("--ckpt", type=str, default="runs/edd_midas.pt")
    ap.add_argument("--num_workers", type=int, default=0)
    return ap.parse_args()

def overfit_one_batch(model, loader, device, steps=200, lr=1e-2):
    model.train()
    batch = next(iter(loader))
    img = batch["image"].to(device); depth = batch["depth"].to(device); mask = batch["mask"].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for t in range(steps):
        pred = model(img)
        loss = silog_loss(pred, depth, mask)  # simple et stable
        opt.zero_grad(); loss.backward(); opt.step()
        if (t+1)%20==0:
            print(f"[overfit] step {t+1} loss={loss.item():.4f}")

def train_one_epoch(model, loader, optimizer, device, scheduler=None):
    model.train(); total = 0.0
    ref_name, ref_w = None, None

    for i, batch in enumerate(loader):
        img   = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        mask  = batch["mask"].to(device, non_blocking=True)

        pred = model(img)  # logits = log(depth)

        if i == 0:
            with torch.no_grad():
                ps = pred[:1]
                print("[dbg] logD mean:", float(ps.mean()),
                      "target mean:", float(depth[:1].mean()),
                      "mask%:", float(mask.mean()))
                # dump visus
                try:
                    import numpy as np, imageio.v2 as imageio
                    x = img[0].detach().cpu().numpy().transpose(1,2,0)
                    x = (x * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
                    x = np.clip(x*255, 0, 255).astype(np.uint8)
                    y = depth[0,0].detach().cpu().numpy()
                    p = pred[0,0].detach().cpu().numpy()
                    p_lin = np.exp(p)
                    y_viz = (np.clip(y/np.percentile(y,99), 0, 1)*255).astype(np.uint8)
                    p_viz = (np.clip(p_lin/np.percentile(p_lin,99), 0, 1)*255).astype(np.uint8)
                    imageio.imwrite("runs/figures/rgb.png",  x)
                    imageio.imwrite("runs/figures/gt.png",   y_viz)
                    imageio.imwrite("runs/figures/pred.png", p_viz)
                    print("[dump] runs/figures/{rgb,gt,pred}.png")
                except Exception as e:
                    print("[dump skipped]", repr(e))
            # paramètre de référence pour delta|w|
            for n, p0 in model.named_parameters():
                if p0.requires_grad and p0.dim() >= 2:
                    ref_name, ref_w = n, p0.detach().clone()
                    print("[ref]", n, tuple(p0.shape))
                    break

        valid = mask.sum() / mask.numel()
        if valid < 0.05:
            print(f"[warn] low valid ratio: {float(valid):.3f}")

        p_lin = torch.exp(pred)
        loss = 0.5 * silog_loss(pred, depth, mask) + 0.5 * ((mask * (p_lin - depth).abs()).sum() / (mask.sum() + 1e-6))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"[grad] ||g||={float(gn):.3e}" if i == 0 else "", end="")

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if i == 0 and ref_name is not None:
            w = dict(model.named_parameters())[ref_name].detach()
            dn = (w - ref_w).abs().mean().item()
            print(f"  [delta] {ref_name} mean|Δw|={dn:.3e}")
            ref_w = w.clone()

        total += loss.item()
    return total / max(1, len(loader))





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

    train_loader, val_loader = make_loaders(
        args.data_root, img_size=args.img_size, batch_size=args.batch_size, limit_samples=args.limit_samples
    )

    model = DPTSmall(backbone_name=args.backbone, pretrained=True).to(device)

    if False:
        overfit_one_batch(model, train_loader, device, steps=200, lr=1e-2)
        return

    
    
    optim = AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    sched  = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=3e-4)


    best = 9e9
    print("batches:", len(train_loader), len(val_loader))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optim, device, scheduler=sched)
        va, met = validate(model, val_loader, device)
        print(f"[epoch {epoch}] train={tr:.4f} val={va:.4f} AbsRel={met['AbsRel']:.4f} RMSE={met['RMSE']:.4f} d1.25={met['Delta<1.25']:.4f} time={time.time()-t0:.1f}s")
        print(f"[epoch {epoch}] lr={sched.get_last_lr()[0]:.2e} train={tr:.4f} val={va:.4f} ...")
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.ckpt)
            print(f"[ckpt] saved -> {args.ckpt}")
    print("[done]")



if __name__ == "__main__":
    main()
