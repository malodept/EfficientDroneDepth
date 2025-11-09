# src/edd/modeling.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DPTSmall(nn.Module):
    """
    Compact DPT-like encoder-decoder using timm backbone.
    """
    def __init__(self, backbone_name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, features_only=True, pretrained=pretrained, out_indices=(1,2,3,4))
        chs = self.backbone.feature_info.channels()
        self.lateral4 = nn.Conv2d(chs[3], 256, 1)
        self.lateral3 = nn.Conv2d(chs[2], 256, 1)
        self.lateral2 = nn.Conv2d(chs[1], 256, 1)
        self.lateral1 = nn.Conv2d(chs[0], 128, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth2 = nn.Conv2d(256, 128, 3, padding=1)
        self.smooth1 = nn.Conv2d(128, 64, 3, padding=1)
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        # encoder
        c1, c2, c3, c4 = self.backbone(x)

        # FPN
        p4 = self.lateral4(c4)
        p3 = F.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False) + self.lateral3(c3)
        p3 = self.smooth3(p3)

        p2 = F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False) + self.lateral2(c2)
        p2 = self.smooth2(p2)

        p1 = F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False) + self.lateral1(c1)  # lateral1 has 128 out-ch
        p1 = self.smooth1(p1)

        # head
        out = self.head(F.interpolate(p1, scale_factor=2, mode="bilinear", align_corners=False))
        H, W = x.shape[-2:]  # force exact input size
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out

def _to_chw(x):
    # (B,H,W,C) -> (B,C,H,W)
    if x.dim() == 4 and x.size(1) not in (1,3,4) and x.size(-1) in (1,3,4):
        x = x.permute(0, 3, 1, 2)
    return x

def _align(pred, target, mask, eps=1e-6):
    pred   = _to_chw(pred)
    target = _to_chw(target)
    mask   = _to_chw(mask)

    if pred.dim()==4 and pred.size(1)==1:   pred   = pred[:,0]
    if target.dim()==4 and target.size(1)==1: target = target[:,0]
    if mask.dim()==4 and mask.size(1)==1:     mask   = mask[:,0]

    if pred.shape[-2:] != target.shape[-2:]:
        target = F.interpolate(target.unsqueeze(1), size=pred.shape[-2:], mode="nearest")[:,0]
        mask   = F.interpolate(mask.unsqueeze(1),   size=pred.shape[-2:], mode="nearest")[:,0]

    target = target.clamp_min(eps)
    return pred, target, mask

def silog_loss(pred_log, target, mask):
    log_d = pred_log
    log_gt = torch.log(torch.clamp(target, min=1e-3))
    diff = (log_d - log_gt)[mask > 0]
    if diff.numel() == 0:
        return torch.tensor(0., device=pred_log.device)
    var = diff.var()
    mean = diff.mean()
    return torch.sqrt(var + 0.15 * mean**2)



def l1_masked(pred, target, mask, eps=1e-6):
    pred, target, mask = _align(pred, target, mask, eps)
    # L1 sur log-profondeur pour la stabilité
    return (mask * (pred - target.log()).abs()).sum() / (mask.sum() + eps)


@torch.no_grad()
def depth_metrics(pred_lin, target, mask, eps=1e-6):
    """
    pred_lin, target: profondeurs en mètres (B,1,H,W)
    mask: (B,1,H,W) binaire {0,1}
    """
    m = (mask > 0.5)
    if not torch.any(m):
        return {"AbsRel": float("nan"), "RMSE": float("nan"), "Delta<1.25": 0.0}

    p = pred_lin[m]
    g = target[m]
    g = torch.clamp(g, min=eps)
    p = torch.clamp(p, min=eps)

    abs_rel = torch.mean(torch.abs(p - g) / g)
    rmse    = torch.sqrt(torch.mean((p - g) ** 2))
    ratio   = torch.max(p / g, g / p)
    d125    = torch.mean((ratio < 1.25).float())

    return {"AbsRel": float(abs_rel), "RMSE": float(rmse), "Delta<1.25": float(d125)}




