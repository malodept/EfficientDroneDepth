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
        c1, c2, c3, c4 = self.backbone(x)
        p4 = self.lateral4(c4)
        p3 = F.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False) + self.lateral3(c3); p3 = self.smooth3(p3)
        p2 = F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False) + self.lateral2(c2); p2 = self.smooth2(p2)
        p1 = F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False) + self.lateral1(c1); p1 = self.smooth1(p1)
        out = self.head(F.interpolate(p1, scale_factor=2, mode="bilinear", align_corners=False))
        return F.relu(out)

def silog_loss(pred, target, mask, lam=0.85):
    eps = 1e-6
    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=eps)
    d = (pred.log() - target.log()) * mask
    n = mask.sum() + eps
    return (d.pow(2).sum()/n) - lam * (d.sum().pow(2) / (n*n))

def l1_masked(pred, target, mask):
    return (torch.abs(pred - target) * mask).sum() / (mask.sum() + 1e-6)

@torch.no_grad()
def depth_metrics(pred, target, mask):
    eps = 1e-6
    m = mask.bool()
    p = pred[m].clamp(min=eps); t = target[m].clamp(min=eps)
    abs_rel = (torch.abs(p - t) / t).mean().item()
    rmse = torch.sqrt(((p - t).pow(2)).mean()).item()
    ratio = torch.max(p/t, t/p)
    delta = (ratio < 1.25).float().mean().item()
    return {"AbsRel": abs_rel, "RMSE": rmse, "Delta<1.25": delta}
