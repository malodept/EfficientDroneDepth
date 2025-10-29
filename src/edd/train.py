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
import imageio
from torch.amp import autocast

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
    ref_name, ref_w = None, None

    for i, batch in enumerate(loader):
        img   = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        mask  = batch["mask"].to(device, non_blocking=True)

        # forward + losses en AMP
        with autocast(device_type="cuda", dtype=torch.float16):
            pred_log = model(img)                           # log(depth) côté réseau
            pred_log = torch.clamp(pred_log, -4.0, 4.0)
            pred_lin = torch.exp(pred_log)                  # repasse en linéaire pour les termes linéaires
            # pertes cohérentes d'échelle
            loss = (
                0.6*silog_loss(pred_log, depth, mask)       # silog attend log vs lin → interne gère log(depth)
                + 0.2*l1_masked(pred_lin, depth, mask)      # L1 en linéaire
                + 0.2*grad_loss_log(pred_log, depth, mask)  # gradient en log
            ).float()

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
        # scheduler: step par ÉPOQUE (pas ici)

        total += float(loss.detach().item())
        # libère le graphe au plus tôt
        del loss, pred_log, pred_lin

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
    metr_sum = None
    n = 0
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
        for batch in loader:
            img   = batch["image"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            mask  = batch["mask"].to(device, non_blocking=True)
            pred_log = torch.clamp(model(img), -4.0, 4.0)
            pred_lin = torch.exp(pred_log)
            mdict    = depth_metrics(pred_lin, depth, mask)  # métriques en linéaire
            if metr_sum is None:
                metr_sum = {k: 0.0 for k in mdict}
            for k,v in mdict.items():
                metr_sum[k] += float(v)
            total += float(silog_loss(pred_log, depth, mask).float().item())
            n += 1
    n = max(1, n)
    metr_avg = {k: v / n for k, v in metr_sum.items()} if metr_sum is not None else {}
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

        # step scheduler en fin d'époque
        if sched is not None:
            sched.step()
    
    print("[done]")





if __name__ == "__main__":
    main()
