import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F

from networks import UNet
from dataset import StainDataset
from metric import fid

def load_models(method: str, device: torch.device):
    """
    Подгружает G и F из папки models/run_<method>
    """
    base = Path(f"models/run_{method}")
    G = UNet().to(device)
    Fnet = UNet().to(device)
    G.load_state_dict(torch.load(base/"G_XtoY.pt", map_location=device))
    Fnet.load_state_dict(torch.load(base/"F_YtoX.pt", map_location=device))
    G.eval(); Fnet.eval()
    return G, Fnet

def visualize_samples(G, Fnet, he_dir, ki_dir, split_path, vis_dir, n_samples):
    os.makedirs(vis_dir, exist_ok=True)
    device = next(G.parameters()).device

    # датасет + тестовый сабсет
    ds = StainDataset(
        he_dir=Path(he_dir),
        ki_dir=Path(ki_dir),
        he_filtered_dir=Path("data/HE_filtered"),
        ki_filtered_dir=Path("data/Ki67_filtered"),
        save_filtered=False
    )
    splits = json.load(open(split_path))
    test_ds = Subset(ds, splits["test"])
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (he_img, ki_img) in enumerate(loader):
            if i >= n_samples:
                break
            he = he_img.to(device)
            ki = ki_img.to(device)

            fake_y = G(he); rec_x = Fnet(fake_y)
            fake_x = Fnet(ki); rec_y = G(fake_x)

            # собираем в grid: [HE, fakeY, recX, Ki, fakeX, recY]
            imgs = torch.cat([he, fake_y, rec_x, ki, fake_x, rec_y], dim=0)
            grid = make_grid(imgs, nrow=3, padding=2, pad_value=1.0)

            out_path = Path(vis_dir)/f"{i:03d}.png"
            save_image(grid, out_path)
            print(f"[VIS] saved {out_path}")

def test_cycle_consistency(G, Fnet, he_dir, ki_dir, split_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = next(G.parameters()).device

    ds = StainDataset(
        he_dir=Path(he_dir),
        ki_dir=Path(ki_dir),
        he_filtered_dir=Path("data/HE_filtered"),
        ki_filtered_dir=Path("data/Ki67_filtered"),
        save_filtered=False
    )
    splits = json.load(open(split_path))
    test_ds = Subset(ds, splits["test"])
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

    total_x = total_y = 0.0
    count = 0
    with torch.no_grad():
        for i, (he_img, ki_img) in enumerate(loader):
            he = he_img.to(device)
            ki = ki_img.to(device)

            fy = G(he); rx = Fnet(fy)
            fx = Fnet(ki); ry = G(fx)

            loss_x = F.l1_loss(rx, he).item()
            loss_y = F.l1_loss(ry, ki).item()

            total_x += loss_x
            total_y += loss_y
            count += 1

            # сохраняем шестерку изображений
            grid = make_grid(torch.cat([he,fy,rx,ki,fx,ry],dim=0), nrow=3, padding=2, pad_value=1.0)
            vutils.save_image(grid, Path(out_dir)/f"cycle_{i:03d}.png")

    mean_x = total_x / count
    mean_y = total_y / count
    mean_all = (mean_x + mean_y) / 2
    print("=== Cycle-Consistency ===")
    print(f"Mean L1 X&rarr;G&rarr;F&rarr;X: {mean_x:.4f}")
    print(f"Mean L1 Y&rarr;F&rarr;G&rarr;Y: {mean_y:.4f}")
    print(f"Overall cycle L1: {mean_all:.4f}")
    return mean_all

def compute_fid_is(G, he_dir, ki_dir, split_path):
    device = next(G.parameters()).device

    ds = StainDataset(
        he_dir=Path(he_dir),
        ki_dir=Path(ki_dir),
        he_filtered_dir=Path("data/HE_filtered"),
        ki_filtered_dir=Path("data/Ki67_filtered"),
        save_filtered=False
    )
    splits = json.load(open(split_path))
    test_ds = Subset(ds, splits["test"])
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

    real, fake = [], []
    with torch.no_grad():
        for he_img, ki_img in loader:
            he = he_img.to(device)
            fy = G(he)
            fake.append(fy.squeeze(0).cpu())
            real.append(ki_img.squeeze(0).cpu())

    fid_value = fid(fake, real)
    print("=== Distribution Metric ===")
    print(f"FID Score:       {fid_value:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Universal test for CycleGAN variants")
    p.add_argument("--method", choices=["adam","ogda","opt_adam","eg"], required=True,
                   help="Which optimizer variant to test")
    p.add_argument("--he-dir",  default="data/dataset_HE/tiles")
    p.add_argument("--ki-dir",  default="data/dataset_Ki67/tiles")
    p.add_argument("--split",   default="split_indices.json")
    p.add_argument("--vis-dir", default="test_visuals")
    p.add_argument("--res-dir", default="test_results")
    p.add_argument("--n-samples", type=int, default=10)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, Fnet = load_models(args.method, device)

    # 1) Cycle-consistency + визуализации
    cycle_out = Path(args.res_dir)/args.method/"cycle"
    vis_out   = Path(args.vis_dir)/args.method
    test_cycle_consistency(G, Fnet, args.he_dir, args.ki_dir, args.split, cycle_out)
    visualize_samples(G, Fnet, args.he_dir, args.ki_dir, args.split, vis_out, args.n_samples)

    # 2) FID и Inception Score (только G&rarr;Y)
    compute_fid_is(G, args.he_dir, args.ki_dir, args.split)