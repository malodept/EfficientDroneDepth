# src/edd/train.py
import os, time, argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR
from rich import print
from .data import make_loaders
from .modeling import DPTSmall, silog_loss, l1_masked, depth_metrics, _align

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

def _grad_xy(x):
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]

def grad_loss_log(pred, target, mask, eps=1e-6):
    # log-espace déjà : pred ~ log(D), target_log = log(target)
    p, t, m = _align(pred, target, mask, eps)  
    gx_p, gy_p = _grad_xy(p)
    gx_t, gy_t = _grad_xy(t.log())
    gx_m, gy_m = _grad_xy(m)
    l = (gx_m * (gx_p - gx_t).abs()).sum() + (gy_m * (gy_p - gy_t).abs()).sum()
    v = gx_m.sum() + gy_m.sum() + eps
    return l / v


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
                # stats logits
                print("[dbg] pred min/median/max:",
                      float(pred.min()), float(pred.median()), float(pred.max()))
                # dump visus sécurisé
                try:
                    import numpy as np, imageio.v2 as imageio
                    x = img[0].detach().cpu().numpy().transpose(1,2,0)
                    x = (x * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
                    x = np.clip(x*255, 0, 255).astype(np.uint8)
                    y = depth[0,0].detach().cpu().numpy()
                    p = pred[0,0].detach().cpu().numpy()
                    p_lin = np.exp(np.clip(p, -10, 10))  # clamp anti-overflow
                    den_y = float(max(1e-6, np.percentile(y,     99)))
                    den_p = float(max(1e-6, np.percentile(p_lin, 99)))
                    y_viz = (np.clip(y/den_y,     0, 1)*255).astype(np.uint8)
                    p_viz = (np.clip(p_lin/den_p, 0, 1)*255).astype(np.uint8)
                    imageio.imwrite("runs/figures/rgb.png",  x)
                    imageio.imwrite("runs/figures/gt.png",   y_viz)
                    imageio.imwrite("runs/figures/pred.png", p_viz)
                    print("[dump] runs/figures/{rgb,gt,pred}.png")
                except Exception as e:
                    print("[dump skipped]", repr(e))
            for n, p0 in model.named_parameters():
                if p0.requires_grad and p0.dim() >= 2:
                    ref_name, ref_w = n, p0.detach().clone()
                    print("[ref]", n, tuple(p0.shape))
                    break

        # gardes validité masque
        valid_pix = mask.sum()
        if valid_pix < 1:
            print("[skip] no valid pixels"); 
            continue

        valid_ratio = valid_pix / mask.numel()
        if valid_ratio < 0.05:
            print(f"[warn] low valid ratio: {float(valid_ratio):.3f}")

        # pertes stables en log-espace (pas d'exp ici)
        loss = 0.6 * silog_loss(pred, depth, mask) \
             + 0.2 * l1_masked(pred, depth, mask) \
             + 0.2 * grad_loss_log(pred, depth, mask)

        if not torch.isfinite(loss):
            print("[skip] non-finite loss"); 
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            print("[skip] non-finite grad"); 
            continue

        if i == 0:
            print(f"[grad] ||g||={float(gn):.3e}", end="")

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if i == 0 and ref_name is not None:
            w = dict(model.named_parameters())[ref_name].detach()
            dn = (w - ref_w).abs().mean().item()
            print(f"  [delta] {ref_name} mean|Δw|={dn:.3e}")
            ref_w = w.clone()

        total += float(loss.item())
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

    backbone_params, head_params = [], []
    for n,p in model.named_parameters():
        (backbone_params if "backbone" in n else head_params).append(p)

    optim = AdamW(
        [{"params": backbone_params, "lr": 2e-4},
         {"params": head_params,     "lr": 1e-3}],
        weight_decay=5e-3
    )
    sched = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=2e-4)

    best = 9e9
    print("batches:", len(train_loader), len(val_loader))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optim, device, scheduler=sched)
        if epoch <= 3:
            for g in optim.param_groups:
                if g["lr"] > 2e-4:
                    g["lr"] = 5e-4
        va, met = validate(model, val_loader, device)
        lrs = [f"{g['lr']:.2e}" for g in optim.param_groups]
        print(f"[epoch {epoch}] lrs={lrs} train={tr:.4f} val={va:.4f} "
              f"AbsRel={met['AbsRel']:.4f} RMSE={met['RMSE']:.4f} d1.25={met['Delta<1.25']:.4f} "
              f"time={time.time()-t0:.1f}s")
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.ckpt)
            print(f"[ckpt] saved -> {args.ckpt}")
    print("[done]")




if __name__ == "__main__":
    main()
