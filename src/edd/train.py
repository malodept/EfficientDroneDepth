# src/edd/train.py
import os, time, argparse
import torch
from pathlib import Path
import csv
import random
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
from rich import print
from .data import make_loaders
from .modeling import DPTSmall, silog_loss, l1_masked, depth_metrics, _align
import imageio
from torch.amp import autocast
import numpy as np  # ensure np is available in validate debug

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
    return x[...,1:,:]-x[...,:-1,:], x[...,:,1:]-x[...,:,:-1]

def grad_loss_log(pred, target, mask, eps=1e-6):
    # aligne et filtre
    p, t, m = _align(pred, target, mask, eps)
    # évite log(0)
    t = torch.clamp(t, min=1e-3)
    # érode le masque pour ne garder que des paires valides
    mx = m[...,1:,:] * m[...,:-1,:]
    my = m[...,:,1:] * m[...,:,:-1]
    # gradients
    gx_p, gy_p = _grad_xy(p)
    gx_t, gy_t = _grad_xy(torch.log(t))
    # L1 pondérée par le masque
    num = (mx * (gx_p - gx_t).abs()).sum() + (my * (gy_p - gy_t).abs()).sum()
    den = mx.sum() + my.sum() + eps
    return num / den


# --- train_one_epoch ---------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, scheduler=None, val_loader=None, scaler=None):
    import os, torch, numpy as np, imageio.v2 as imageio
    model.train()
    total = 0.0
    # dossier debug (par epoch)
    dbg_dir = Path("runs/debug") / f"epoch_{int(time.time())}"
    dbg_dir.mkdir(parents=True, exist_ok=True)
    # logger CSV des pertes (header)
    csv_path = dbg_dir / "batch_losses.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "loss", "silog", "l1_lin", "grad_log", "gnorm"])

    for i, batch in enumerate(loader):
        img   = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        mask  = batch["mask"].to(device, non_blocking=True)

        # forward + pertes en AMP
        with autocast(device_type="cuda", dtype=torch.float16):
            pred_log = model(img)                    # log(depth)
            pred_log = torch.clamp(pred_log, -4.0, 4.0)
            pred_lin = torch.exp(pred_log)           # pour les termes en linéaire
            # décompose les pertes pour logging
            sil = silog_loss(pred_log, depth, mask)
            l1  = l1_masked(pred_lin,  depth, mask)
            grd = grad_loss_log(pred_log, depth, mask)
            loss = (0.6*sil + 0.2*l1 + 0.2*grd).float()

        # --- debug first batch only, SANS GRAD ---
        if i == 0:
            with torch.no_grad():
                os.makedirs("runs/figures", exist_ok=True)
                ps = pred_log[:1,0].detach()
                print("[dbg] logD mean:", float(ps.mean()),
                      "target mean:", float(depth[:1].mean().detach()),
                      "mask%:", float(mask.mean().detach()))
                print("[dbg] pred min/median/max:",
                      float(pred_log.min().detach()), float(pred_log.median().detach()), float(pred_log.max().detach()))
                try:
                    x = img[0].detach().cpu().numpy().transpose(1,2,0)
                    x = (x * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
                    x = np.clip(x*255, 0, 255).astype(np.uint8)
                    y = depth[0,0].detach().cpu().numpy()
                    p = pred_log[0,0].detach().cpu().numpy()
                    p_lin = np.exp(np.clip(p, -10, 10))
                    y_viz = (np.clip(y/ max(1e-6, np.percentile(y,99)),0,1)*255).astype(np.uint8)
                    p_viz = (np.clip(p_lin/ max(1e-6, np.percentile(p_lin,99)),0,1)*255).astype(np.uint8)
                    imageio.imwrite("runs/figures/rgb.png",  x)
                    imageio.imwrite("runs/figures/gt.png",   y_viz)
                    imageio.imwrite("runs/figures/pred.png", p_viz)
                except Exception as e:
                    print("[dbg] viz error:", e)
            # Histogrammes rapides & scatter pour le batch 0
            with torch.no_grad():
                def _save_hist(arr, title, fn):
                    arr = arr.flatten().detach().float().cpu().numpy()
                    plt.figure(figsize=(4,3))
                    plt.hist(arr, bins=80)
                    plt.title(title)
                    plt.tight_layout()
                    plt.savefig(fn); plt.close()
                _save_hist(pred_log[:1].float(), "pred_log", dbg_dir/"hist_pred_log_b0.png")
                _save_hist(pred_lin[:1].float(), "pred_lin", dbg_dir/"hist_pred_lin_b0.png")
                _save_hist(depth[:1].float(),    "depth_lin", dbg_dir/"hist_depth_lin_b0.png")
                # scatter calibrage
                x = depth[:1,0].flatten().detach().float().cpu().numpy()
                y = pred_lin[:1,0].flatten().detach().float().cpu().numpy()
                m = mask[:1,0].flatten().detach().float().cpu().numpy() > 0.5
                x, y = x[m][::10], y[m][::10]
                plt.figure(figsize=(4,4)); plt.scatter(x, y, s=4)
                plt.xlabel("GT depth"); plt.ylabel("Pred depth")
                lim = (0, max(1e-6, np.percentile(np.r_[x,y], 99)))
                plt.xlim(lim); plt.ylim(lim); plt.plot(lim, lim)
                plt.tight_layout(); plt.savefig(dbg_dir/"scatter_gt_vs_pred_b0.png"); plt.close()

        optimizer.zero_grad(set_to_none=True)
        # backward unique + clipping 0.1
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
        # log CSV (append)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            lr0 = optimizer.param_groups[0]["lr"]
            try:
                gn_val = float(gn)
            except Exception:
                gn_val = float(gn.detach().item()) if hasattr(gn, "detach") else 0.0
            w.writerow([i, f"{lr0:.3e}", float(loss.detach().item()),
                        float(sil.detach().item()), float(l1.detach().item()), float(grd.detach().item()),
                        gn_val])
        # scheduler: step par ÉPOQUE (pas ici)

        total += float(loss.detach().item())
        # libère le graphe au plus tôt
        del loss, pred_log, pred_lin, sil, l1, grd

    # validation: log du %valide
    if val_loader is not None:
        with torch.no_grad():
            val_valid = 0.0
            for batch in val_loader:
                val_valid += batch["mask"].to(device).mean().item()
            val_valid /= max(1, len(val_loader))
            print(f"[val] valid%≈{val_valid:.3f}")

    return total / max(1, len(loader))










# --- validate ---------------------------------------------------------------
def validate(model, loader, device):
    model.eval()
    total = 0.0
    metr_sum = defaultdict(float)   # accumulate safely
    n = 0                           # number of valid batches for metrics
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
        for batch in loader:
            img   = batch["image"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            mask  = batch["mask"].to(device, non_blocking=True)
            pred_log = torch.clamp(model(img), -4.0, 4.0)
            pred_lin = torch.exp(pred_log)
            mdict    = depth_metrics(pred_lin, depth, mask)  # métriques en linéaire
            # ignore empty/None metrics
            if not mdict:
                total += float(silog_loss(pred_log, depth, mask).float().item())
                continue
            # dump une seule fois sur la première itération val
            if n == 0:
                vd = Path("runs/debug/val")
                vd.mkdir(parents=True, exist_ok=True)
                # stats de biais d’échelle: y ≈ s*x
                x = depth[:,0][mask[:,0] > 0.5].detach().float().cpu().numpy()
                y = pred_lin[:,0][mask[:,0] > 0.5].detach().float().cpu().numpy()
                x_, y_ = x[::50], y[::50]
                if x_.size > 10 and y_.size > 10:
                    plt.figure(figsize=(4,4)); plt.scatter(x_, y_, s=4)
                    try:
                        lim_max = max(1e-6, float(np.percentile(np.r_[x_, y_], 99)))
                    except Exception:
                        lim_max = float(max(1e-6, x_.max() if x_.size else 1.0, y_.max() if y_.size else 1.0))
                    lim = (0, lim_max)
                    plt.xlim(lim); plt.ylim(lim); plt.plot(lim, lim)
                    plt.xlabel("GT depth"); plt.ylabel("Pred depth")
                    plt.tight_layout(); plt.savefig(vd/"scatter_val_gt_vs_pred.png"); plt.close()
                # histos
                for arr, name in [(pred_log, "pred_log"), (pred_lin, "pred_lin"), (depth, "depth_lin")]:
                    a = arr.detach().float().cpu().numpy().ravel()[::50]
                    plt.figure(figsize=(4,3)); plt.hist(a, bins=80); plt.title(name)
                    plt.tight_layout(); plt.savefig(vd/("hist_"+name+".png")); plt.close()

            # accumulate only finite values
            for k, v in mdict.items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isfinite(fv):
                    metr_sum[k] += fv
            total += float(silog_loss(pred_log, depth, mask).float().item())
            n += 1
    n = max(1, n)
    metr_avg = {k: v / n for k, v in metr_sum.items()} if len(metr_sum) else {}
    return total / n, metr_avg


# --- main -------------------------------------------------------------------
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

    # optimizer + scheduler
    optim = AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)

    # GradScaler (PyTorch >=2.2)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    
    # scheduler per-epoch
    sched = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-5)

    # helpers: freeze/unfreeze backbone au démarrage
    def set_backbone_trainable(net, flag: bool):
        for n,p in net.named_parameters():
            if n.startswith("backbone."):
                p.requires_grad = flag

    best = 9e9
    print("batches:", len(train_loader), len(val_loader))

    for epoch in range(args.epochs):
        # freeze epoch 0, unfreeze ensuite
        if epoch == 0:
            set_backbone_trainable(model, False)
        elif epoch == 1:
            set_backbone_trainable(model, True)

        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optim, device,
                             scheduler=sched, val_loader=val_loader, scaler=scaler)
        va, met = validate(model, val_loader, device)

        lrs = [f"{g['lr']:.2e}" for g in optim.param_groups]
        print(f"[epoch {epoch+1}] lrs={lrs} train={tr:.4f} val={va:.4f} "
              f"AbsRel={met['AbsRel']:.4f} RMSE={met['RMSE']:.4f} d1.25={met['Delta<1.25']:.4f} "
              f"time={time.time()-t0:.1f}s")

        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.ckpt)
            print(f"[ckpt] saved -> {args.ckpt}")

        # step scheduler once per epoch (after validation and potential ckpt)
        if sched is not None:
            sched.step()
    
    # debug CSVs are under runs/debug
    print(f"[debug] Wrote batch losses under: runs/debug")
    print("[done]")





if __name__ == "__main__":
    main()
