from pathlib import Path
import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from cycle_loss import CycleGANLoss
from networks import UNet, VGGDiscriminator
from dataset import StainDataset
from algos import EG


class CycleGANTrainer:
    def __init__(
        self,
        G_XtoY,
        F_YtoX,
        D_X,
        D_Y,
        dataset,
        *,
        batch_size: int = 8,
        lr_g: float = 2e-4,
        lr_d: float = 1e-5,
        weight_decay: float = 0,
        lambda_cycle: float = 10.0,
        grad_clip_value: float | None = None,
        epochs: int = 100,
        save_every: int = 10
    ):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.save_every = save_every
        self.grad_clip_value = grad_clip_value

        self.G_XtoY = G_XtoY.to(self.device)
        self.F_YtoX = F_YtoX.to(self.device)
        self.D_X = D_X.to(self.device)
        self.D_Y = D_Y.to(self.device)

        g = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = random_split(dataset, [0.6, 0.2, 0.2], g)

        split_dict = {
            "train": list(train_ds.indices),
            "val":   list(val_ds.indices),
            "test":  list(test_ds.indices),
        }
        with open("split_indices.json", "w") as f:
            json.dump(split_dict, f, indent=2)

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        # --- Extra‑Gradient оптимизаторы ---------------------------------- #
        #  генераторы – минимизация; дискриминаторы – максимизация
        self.opt_g  = EG(
            list(self.G_XtoY.parameters()) + list(self.F_YtoX.parameters()),
            lr=lr_g,
            weight_decay=weight_decay,
            maximize=False,
        )
        self.opt_dx = EG(
            self.D_X.parameters(),
            lr=lr_d,
            weight_decay=weight_decay,
            maximize=True,
        )
        self.opt_dy = EG(
            self.D_Y.parameters(),
            lr=lr_d,
            weight_decay=weight_decay,
            maximize=True,
        )

        # --- функции потерь ---------------------------------------------- #
        self.cycle_loss_fn = CycleGANLoss(lambda_cycle=lambda_cycle)
        self.mse = nn.MSELoss()

        self.logger = SummaryWriter()

    # -------------------------------------------------------------------- #
    # training helpers
    # -------------------------------------------------------------------- #
    def _maybe_clip(self, params):
        """Клиппинг градиентов (если grad_clip_value задан)."""
        if self.grad_clip_value is not None:
            utils.clip_grad_norm_(params, self.grad_clip_value)

    # ===========================  ONE EPOCH  ============================ #
    def train_epoch(self):
        self.G_XtoY.train()
        self.F_YtoX.train()
        self.D_X.train()
        self.D_Y.train()

        sum_g = sum_dx = sum_dy = 0.0
        steps = 0

        for real_x, real_y in tqdm(self.train_loader, desc="Train"):
            real_x, real_y = real_x.to(self.device), real_y.to(self.device)

            # ------------------ 1) Discriminator X ------------------------ #
            fake_x_det = self.F_YtoX(real_y).detach()

            def dx_closure():
                self.opt_dx.zero_grad()

                dx_real = self.D_X(real_x)
                dx_fake = self.D_X(fake_x_det)

                loss = 0.5 * (
                    self.mse(dx_real, torch.ones_like(dx_real)) +
                    self.mse(dx_fake, torch.zeros_like(dx_fake))
                )
                loss.backward()
                self._maybe_clip(self.D_X.parameters())
                return loss

            dx_loss = self.opt_dx.step(dx_closure)

            # ------------------ 2) Discriminator Y ------------------------ #
            fake_y_det = self.G_XtoY(real_x).detach()

            def dy_closure():
                self.opt_dy.zero_grad()

                dy_real = self.D_Y(real_y)
                dy_fake = self.D_Y(fake_y_det)

                loss = 0.5 * (
                    self.mse(dy_real, torch.ones_like(dy_real)) +
                    self.mse(dy_fake, torch.zeros_like(dy_fake))
                )
                loss.backward()
                self._maybe_clip(self.D_Y.parameters())
                return loss

            dy_loss = self.opt_dy.step(dy_closure)

            # ------------------ 3) Generators (G & F) --------------------- #
            def g_closure():
                self.opt_g.zero_grad()

                fake_y = self.G_XtoY(real_x)
                rec_x  = self.F_YtoX(fake_y)
                fake_x = self.F_YtoX(real_y)
                rec_y  = self.G_XtoY(fake_x)

                dy_fake_for_g = self.D_Y(fake_y)
                dx_fake_for_f = self.D_X(fake_x)

                loss = self.cycle_loss_fn(
                    real_x, real_y,
                    fake_y, fake_x,
                    rec_x,  rec_y,
                    dy_fake_for_g, dx_fake_for_f,
                )
                loss.backward()
                self._maybe_clip(
                    list(self.G_XtoY.parameters()) + list(self.F_YtoX.parameters())
                )
                return loss

            g_loss = self.opt_g.step(g_closure)

            # --------- накопление статистики ----------------------------- #
            sum_g  += g_loss.item()
            sum_dx += dx_loss.item()
            sum_dy += dy_loss.item()
            steps  += 1

        return sum_g / steps, sum_dx / steps, sum_dy / steps

    # ===========================  VALIDATION  =========================== #
    @torch.no_grad()
    def validate(self):
        self.G_XtoY.eval()
        self.F_YtoX.eval()
        self.D_X.eval()
        self.D_Y.eval()

        sum_g = sum_dx = sum_dy = 0.0
        steps = 0

        for real_x, real_y in tqdm(self.val_loader, desc="Val"):
            real_x, real_y = real_x.to(self.device), real_y.to(self.device)

            # forward
            fake_y = self.G_XtoY(real_x)
            rec_x  = self.F_YtoX(fake_y)
            fake_x = self.F_YtoX(real_y)
            rec_y  = self.G_XtoY(fake_x)

            # дискриминаторы
            dx_real = self.D_X(real_x)
            dx_fake = self.D_X(fake_x)
            dy_real = self.D_Y(real_y)
            dy_fake = self.D_Y(fake_y)

            dx_loss = 0.5 * (
                self.mse(dx_real, torch.ones_like(dx_real)) +
                self.mse(dx_fake, torch.zeros_like(dx_fake))
            )
            dy_loss = 0.5 * (
                self.mse(dy_real, torch.ones_like(dy_real)) +
                self.mse(dy_fake, torch.zeros_like(dy_fake))
            )

            g_loss = self.cycle_loss_fn(
                real_x, real_y,
                fake_y, fake_x,
                rec_x,  rec_y,
                dy_fake, dx_fake,
            )

            sum_g  += g_loss.item()
            sum_dx += dx_loss.item()
            sum_dy += dy_loss.item()
            steps  += 1

        return sum_g / steps, sum_dx / steps, sum_dy / steps

    # ==========================  CHECKPOINTS  =========================== #
    def _ckpt_path(self, epoch):
        ckpt_dir = Path("models/checkpoints_EG")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir / f"checkpoint_{epoch}.pth"

    def save_checkpoint(self, epoch: int):
        torch.save(
            {
                "epoch": epoch,
                "G_XtoY": self.G_XtoY.state_dict(),
                "F_YtoX": self.F_YtoX.state_dict(),
                "D_X": self.D_X.state_dict(),
                "D_Y": self.D_Y.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_dx": self.opt_dx.state_dict(),
                "opt_dy": self.opt_dy.state_dict(),
            },
            self._ckpt_path(epoch),
        )
        print(f"checkpoint saved (epoch {epoch})")

    # -------------------------------------------------------------------- #
    def run(self):
        for epoch in range(1, self.epochs + 1):
            # ---------- train ------------------------------------------- #
            tr_g, tr_dx, tr_dy = self.train_epoch()
            self.logger.add_scalar("loss/train_g",  tr_g,  epoch)
            self.logger.add_scalar("loss/train_dx", tr_dx, epoch)
            self.logger.add_scalar("loss/train_dy", tr_dy, epoch)

            # ---------- val --------------------------------------------- #
            val_g, val_dx, val_dy = self.validate()
            self.logger.add_scalar("loss/val_g",  val_g,  epoch)
            self.logger.add_scalar("loss/val_dx", val_dx, epoch)
            self.logger.add_scalar("loss/val_dy", val_dy, epoch)

            print(
                f"[{epoch:03d}/{self.epochs}] "
                f"Train G:{tr_g:.4f} DX:{tr_dx:.4f} DY:{tr_dy:.4f} | "
                f"Val G:{val_g:.4f} DX:{val_dx:.4f} DY:{val_dy:.4f}"
            )

            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch)

        print("Training complete!")
        self.logger.close()

if __name__ == "__main__":

    G_XtoY = UNet()
    F_YtoX = UNet()
    D_X = VGGDiscriminator()
    D_Y = VGGDiscriminator()

    he_dir = Path("data/dataset_HE/tiles")
    ki_dir = Path("data/dataset_Ki67/tiles")
    dataset = StainDataset(
        he_dir=he_dir,
        ki_dir=ki_dir,
        he_filtered_dir=Path("data/HE_filtered"),
        ki_filtered_dir=Path("data/Ki67_filtered"),
        save_filtered=False
    )

    trainer = CycleGANTrainer(
        G_XtoY,
        F_YtoX,
        D_X,
        D_Y,
        dataset=dataset,
        batch_size=8,
        lr_g=2e-4,
        lr_d=1e-5,
        weight_decay=0,
        lambda_cycle=10.0,
        grad_clip_value=None,
        epochs=100,
        save_every=10
    )

    trainer.run()

    out_dir = Path("models/run_EG")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.G_XtoY.state_dict(), out_dir / "G_XtoY.pt")
    torch.save(trainer.F_YtoX.state_dict(), out_dir / "F_YtoX.pt")
    torch.save(trainer.D_X.state_dict(),    out_dir / "D_X.pt")
    torch.save(trainer.D_Y.state_dict(),    out_dir / "D_Y.pt")
