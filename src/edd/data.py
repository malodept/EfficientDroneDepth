# src/edd/data.py
import os, glob, random
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

def _resize_pad(arr, size, interp):
    h, w = arr.shape[:2]
    s = size / max(h, w)
    nh, nw = int(h*s), int(w*s)
    arr = cv2.resize(arr, (nw, nh), interpolation=interp)
    pad_h, pad_w = size - nh, size - nw
    arr = cv2.copyMakeBorder(arr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    return arr

def _read_image(path: str, size: int = 384):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _resize_pad(img, size, cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return np.transpose(img, (2, 0, 1))

def _read_depth(path, size):
    d = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if d is None: raise FileNotFoundError(path)
    if d.ndim == 3: d = d[...,0]
    d = _resize_pad(d, size, cv2.INTER_NEAREST).astype(np.float32)
    mx = float(d.max())
    if mx > 5000:   d /= 1000.0   # mm→m
    elif mx <= 255: d /= 100.0    # cm→m
    d = np.clip(d, 0.0, 80.0)
    m = (d > 1e-6).astype(np.float32)
    return d, m



class TartanAirDepth(Dataset):
    """
    Minimal TartanAir dataset loader expecting paired RGB/depth with matching ids.
    """
    def __init__(self, root: str, img_size: int = 384, limit_samples: Optional[int] = None, split_ratio: float = 0.9, train: bool = True):
        self.img_size = img_size
        left_paths = glob.glob(os.path.join(root, "**", "left", "*_left.*"), recursive=True)
        pairs = []
        for lp in left_paths:
            base = os.path.basename(lp).replace("_left", "")
            stem = os.path.splitext(base)[0]
            depth_candidate = os.path.join(os.path.dirname(os.path.dirname(lp)), "depth", f"{stem}_depth.png")
            if os.path.exists(depth_candidate):
                pairs.append((lp, depth_candidate))
        if not pairs:
            raise RuntimeError(f"No pairs found under {root}.")
        random.seed(42); random.shuffle(pairs)
        if limit_samples is not None:
            pairs = pairs[:limit_samples]
        split = int(len(pairs) * split_ratio)
        self.pairs = pairs[:split] if train else pairs[split:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, depth_path = self.pairs[idx]
        img = _read_image(img_path, self.img_size)
        depth_np, mask_np = _read_depth(depth_path, self.img_size)
        depth = torch.from_numpy(depth_np).unsqueeze(0)  # (1,H,W)
        mask  = torch.from_numpy(mask_np ).unsqueeze(0)  # (1,H,W)
        return {
            "image": torch.from_numpy(img).float(),
            "depth": depth.float(),
            "mask": mask.float(),
            "img_path": img_path,
            "depth_path": depth_path,
        }

def make_loaders(root: str, img_size: int = 384, batch_size: int = 8, num_workers: int = 2, limit_samples: Optional[int] = None):
    train_ds = TartanAirDepth(root, img_size=img_size, limit_samples=limit_samples, train=True)
    val_ds = TartanAirDepth(root, img_size=img_size, limit_samples=limit_samples, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
