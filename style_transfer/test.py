import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image
from pathlib import Path
import json
import os
import torchvision.utils as vutils
import torch.nn.functional as F

from networks import UNet
from dataset import StainDataset

def visualize_cycleGAN_samples(
    G_XtoY_path: str, 
    F_YtoX_path: str, 
    he_dir: str, 
    ki_dir: str, 
    split_indices_path="split_indices.json",
    save_dir="test_visuals",
    n_samples=5
):
    """
    Выбираем n_samples из тестовой выборки, 
    генерируем картинки (fake_y, rec_x, fake_x, rec_y) и сохраняем 
    в коллаж (grid) для каждого сэмпла.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Подгруждаем датасет
    dataset = StainDataset(Path(he_dir), Path(ki_dir))
    with open(split_indices_path, "r") as f:
        split_dict = json.load(f)
    test_indices = split_dict["test"]
    test_data = Subset(dataset, test_indices)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    #Подгружаем модели
    G_XtoY = UNet().to(device)
    F_YtoX = UNet().to(device)
    G_XtoY.load_state_dict(torch.load(G_XtoY_path, map_location=device))
    F_YtoX.load_state_dict(torch.load(F_YtoX_path, map_location=device))
    G_XtoY.eval()
    F_YtoX.eval()

    #Цикл по первым n_samples
    with torch.no_grad():
        for i, (he_img, ki_img) in enumerate(test_loader):
            if i >= n_samples:
                break

            he_img = he_img.to(device)
            ki_img = ki_img.to(device)

            # Генерируем
            fake_y = G_XtoY(he_img)   # x->y
            rec_x  = F_YtoX(fake_y)   # x->y->x

            fake_x = F_YtoX(ki_img)   # y->x
            rec_y  = G_XtoY(fake_x)   # y->x->y

            # Создадим коллаж
            # Например, соберём [ real_x, fake_y, rec_x, real_y, fake_x, rec_y ] в одну строку
            # shape: (6, C, H, W)
            collage = torch.cat([he_img, fake_y, rec_x, ki_img, fake_x, rec_y], dim=0)

            # Превращаем в grid (nrow=6 => все в одну строку) 
            # или nrow=3, тогда будет 2 строки, etc.
            # pad_value=1 — белые границы
            grid = make_grid(collage, nrow=3, padding=2, pad_value=1.0)

            # Сохраним
            out_path = os.path.join(save_dir, f"sample_{i}_visual.png")
            save_image(grid, out_path)
            print(f"[INFO] Сохранена визуализация сэмпла {i}: {out_path}")


def test_cycleGAN_with_cycle_consistency(
    G_XtoY_path: str, 
    F_YtoX_path: str, 
    he_dir: str, 
    ki_dir: str, 
    split_indices_path="split_indices.json",
    save_dir="test_results"
):
    """
    Тестируем CycleGAN, дополнительно считая "cycle-consistency" метрику (L1):
      - x -> G -> fake_y -> F -> rec_x, сравниваем rec_x с x
      - y -> F -> fake_x -> G -> rec_y, сравниваем rec_y с y
    Усреднённое L1 показывает, насколько модель хорошо восстанавливает вход после 2 преобразований.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    dataset = StainDataset(Path(he_dir), Path(ki_dir))

    with open(split_indices_path, "r") as f:
        split_dict = json.load(f)
    test_indices = split_dict["test"]
    test_data = Subset(dataset, test_indices)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    G_XtoY = UNet().to(device)
    F_YtoX = UNet().to(device)

    G_XtoY.load_state_dict(torch.load(G_XtoY_path, map_location=device))
    F_YtoX.load_state_dict(torch.load(F_YtoX_path, map_location=device))
    
    G_XtoY.eval()
    F_YtoX.eval()

    total_cycle_x = 0.0
    total_cycle_y = 0.0
    count = 0

    with torch.no_grad():
        for i, (he_img, ki_img) in enumerate(test_loader):
            he_img = he_img.to(device)
            ki_img = ki_img.to(device)

            # (A) X -> G -> Y -> F -> rec_X
            fake_y = G_XtoY(he_img)
            rec_x  = F_YtoX(fake_y)

            # (B) Y -> F -> X -> G -> rec_Y
            fake_x = F_YtoX(ki_img)
            rec_y  = G_XtoY(fake_x)

            # --- Считаем L1 между x и rec_x, y и rec_y
            cycle_x_loss = F.l1_loss(rec_x, he_img)
            cycle_y_loss = F.l1_loss(rec_y, ki_img)

            total_cycle_x += cycle_x_loss.item()
            total_cycle_y += cycle_y_loss.item()
            count += 1

            out_he_path  = os.path.join(save_dir, f"sample_{i}_realHE.png")
            out_ki_path  = os.path.join(save_dir, f"sample_{i}_realKi.png")
            out_fy_path  = os.path.join(save_dir, f"sample_{i}_fakeY.png")
            out_fx_path  = os.path.join(save_dir, f"sample_{i}_fakeX.png")
            out_rx_path  = os.path.join(save_dir, f"sample_{i}_recX.png")
            out_ry_path  = os.path.join(save_dir, f"sample_{i}_recY.png")

            vutils.save_image(he_img, out_he_path)
            vutils.save_image(ki_img, out_ki_path)
            vutils.save_image(fake_y, out_fy_path)
            vutils.save_image(fake_x, out_fx_path)
            vutils.save_image(rec_x,  out_rx_path)
            vutils.save_image(rec_y,  out_ry_path)

    # 4) Усредняем L1 по всем тест-примерам
    avg_cycle_x = total_cycle_x / count if count > 0 else 0.0
    avg_cycle_y = total_cycle_y / count if count > 0 else 0.0
    avg_cycle = (avg_cycle_x + avg_cycle_y)/2

    print("=== Cycle-Consistency Test Results ===")
    print(f"Mean L1 for x->G->F->x: {avg_cycle_x:.4f}")
    print(f"Mean L1 for y->F->G->y: {avg_cycle_y:.4f}")
    print(f"Overall cycle L1: {avg_cycle:.4f}")

    print("Test inference complete!")


if __name__ == "__main__":
    he_dir = Path("data/dataset_HE/tiles")
    ki_dir = Path("data/dataset_Ki67/tiles")
    test_cycleGAN_with_cycle_consistency(
        G_XtoY_path="models/G_XtoY.pt",
        F_YtoX_path="models/F_YtoX.pt",
        he_dir=he_dir,
        ki_dir=ki_dir,
        split_indices_path="split_indices.json",
        save_dir="test_results"
    )
    visualize_cycleGAN_samples(
        G_XtoY_path="models/G_XtoY.pt",
        F_YtoX_path="models/F_YtoX.pt",
        he_dir=he_dir,
        ki_dir=ki_dir,
        split_indices_path="split_indices.json",
        save_dir="test_visuals",
        n_samples=5
    )