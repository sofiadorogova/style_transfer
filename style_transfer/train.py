import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.utils as utils
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from cycle_loss import CycleGANLoss
from dataset import StainDataset
from networks import UNet, VGGDiscriminator

from algos.adam import Adam
from algos.ogda import OGDA
from algos.optimistic_adam import OptimisticAdam
from algos.eg import EG

OPTIMIZERS = {
    "adam": Adam,
    "ogda": OGDA,
    "opt_adam": OptimisticAdam,
    "eg": EG,
}

class CycleGANTrainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.start_epoch = 1
        if args.resume is not None and Path(args.resume).exists():
            ckpt = torch.load(args.resume, map_location="cpu")
            self.start_epoch = ckpt["epoch"] + 1
        # модели
        self.G = UNet().to(self.device)
        self.F = UNet().to(self.device)
        self.Dx = VGGDiscriminator().to(self.device)
        self.Dy = VGGDiscriminator().to(self.device)

        # датасет и split
        ds = StainDataset(
            he_dir=Path(args.he_dir),
            ki_dir=Path(args.ki_dir),
            he_filtered_dir=Path(args.he_filt),
            ki_filtered_dir=Path(args.ki_filt),
            save_filtered=False
        )
        g = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = random_split(ds, [0.6,0.2,0.2], g)
        with open("split_indices.json", "w") as f:
            json.dump({
                "train": list(train_ds.indices),
                "val":   list(val_ds.indices),
                "test":  list(test_ds.indices),
            }, f, indent=2)

        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

        # loss‑функции
        self.cycle_loss_fn = CycleGANLoss(
            lambda_cycle=args.lambda_cycle,
            lambda_id=args.lambda_id
        )
        self.mse = torch.nn.MSELoss()

        # hyperparams
        self.epochs        = args.epochs
        self.save_every    = args.save_every
        self.grad_clip     = args.grad_clip
        self.disc_steps    = args.disc_steps
        self.label_smooth  = args.label_smoothing

        # оптимизаторы
        OptG = OPTIMIZERS[args.optimizer_g]
        OptD = OPTIMIZERS[args.optimizer_d]

        self.opt_g  = OptG(
            list(self.G.parameters()) + list(self.F.parameters()),
            lr=args.lr_g,
            weight_decay=args.weight_decay,
            maximize=False
        )
        self.opt_dx = OptD(
            self.Dx.parameters(),
            lr=args.lr_d,
            weight_decay=args.weight_decay,
            maximize=True
        )
        self.opt_dy = OptD(
            self.Dy.parameters(),
            lr=args.lr_d,
            weight_decay=args.weight_decay,
            maximize=True
        )

        self.logger = SummaryWriter(log_dir=args.logdir)

        if args.resume is not None and Path(args.resume).exists():
            self.G.load_state_dict(ckpt["G"])
            self.F.load_state_dict(ckpt["F"])
            self.Dx.load_state_dict(ckpt["Dx"])
            self.Dy.load_state_dict(ckpt["Dy"])
            self.opt_g.load_state_dict(ckpt["opt_g"])
            self.opt_dx.load_state_dict(ckpt["opt_dx"])
            self.opt_dy.load_state_dict(ckpt["opt_dy"])
            print(f"Loaded checkpoint “{args.resume}” (epoch {ckpt['epoch']})")
 

    def _clip(self, params):
        if self.grad_clip is not None:
            utils.clip_grad_norm_(params, self.grad_clip)

    def train_epoch(self):
        self.G.train(); self.F.train()
        self.Dx.train(); self.Dy.train()

        sum_g = sum_dx = sum_dy = 0.0
        steps = 0

        for x, y in tqdm(self.train_loader, desc="Train"):
            x, y = x.to(self.device), y.to(self.device)

            # --- update discriminators disc_steps times each --- #
            for _ in range(self.disc_steps):
                # D_x
                def closure_dx():
                    self.opt_dx.zero_grad()
                    fake_x = self.F(y).detach()
                    real_out = self.Dx(x)
                    fake_out = self.Dx(fake_x)
                    # label smoothing on real
                    real_target = torch.full_like(real_out, 1.0 - self.label_smooth)
                    loss = 0.5 * (
                        self.mse(real_out, real_target) +
                        self.mse(fake_out, torch.zeros_like(fake_out))
                    )
                    loss.backward()
                    self._clip(self.Dx.parameters())
                    return loss
                dx_loss = self.opt_dx.step(closure_dx)

                # D_y
                def closure_dy():
                    self.opt_dy.zero_grad()
                    fake_y = self.G(x).detach()
                    real_out = self.Dy(y)
                    fake_out = self.Dy(fake_y)
                    real_target = torch.full_like(real_out, 1.0 - self.label_smooth)
                    loss = 0.5 * (
                        self.mse(real_out, real_target) +
                        self.mse(fake_out, torch.zeros_like(fake_out))
                    )
                    loss.backward()
                    self._clip(self.Dy.parameters())
                    return loss
                dy_loss = self.opt_dy.step(closure_dy)

            # --- update generators G & F --- #
            def closure_g():
                self.opt_g.zero_grad()
                fake_y = self.G(x); rec_x = self.F(fake_y)
                fake_x = self.F(y); rec_y = self.G(fake_x)
                same_x = self.F(x); same_y = self.G(y)
                d_y = self.Dy(fake_y); d_x = self.Dx(fake_x)

                loss = self.cycle_loss_fn(
                    real_x=x, real_y=y,
                    fake_y=fake_y, fake_x=fake_x,
                    rec_x=rec_x, rec_y=rec_y,
                    same_x=same_x, same_y=same_y,
                    d_y_fake=d_y, d_x_fake=d_x,
                    is_train_part=True
                )
                loss.backward()
                self._clip(list(self.G.parameters()) + list(self.F.parameters()))
                return loss
            g_loss = self.opt_g.step(closure_g)

            sum_dx += dx_loss.item()
            sum_dy += dy_loss.item()
            sum_g  += g_loss.item()
            steps += 1

        return sum_g/steps, sum_dx/steps, sum_dy/steps

    @torch.no_grad()
    def validate(self):
        self.G.eval(); self.F.eval()
        self.Dx.eval(); self.Dy.eval()

        sum_g = sum_dx = sum_dy = 0.0
        steps = 0

        for x, y in tqdm(self.val_loader, desc="Validate"):
            x, y = x.to(self.device), y.to(self.device)

            fake_y = self.G(x); rec_x = self.F(fake_y)
            fake_x = self.F(y); rec_y = self.G(fake_x)
            same_x = self.F(x); same_y = self.G(y)
            d_y = self.Dy(fake_y); d_x = self.Dx(fake_x)

            # D_x loss
            dx_real = self.Dx(x); dx_fake = self.Dx(fake_x)
            dx_loss = 0.5 * (
                self.mse(dx_real, torch.ones_like(dx_real)) +
                self.mse(dx_fake, torch.zeros_like(dx_fake))
            )
            # D_y loss
            dy_real = self.Dy(y); dy_fake = self.Dy(fake_y)
            dy_loss = 0.5 * (
                self.mse(dy_real, torch.ones_like(dy_real)) +
                self.mse(dy_fake, torch.zeros_like(dy_fake))
            )
            # G/F loss (no identity on val)
            g_loss = self.cycle_loss_fn(
                real_x=x, real_y=y,
                fake_y=fake_y, fake_x=fake_x,
                rec_x=rec_x, rec_y=rec_y,
                same_x=same_x, same_y=same_y,
                d_y_fake=d_y, d_x_fake=d_x,
                is_train_part=False
            )

            sum_g  += g_loss.item()
            sum_dx += dx_loss.item()
            sum_dy += dy_loss.item()
            steps += 1

        return sum_g/steps, sum_dx/steps, sum_dy/steps

    def save_ckpt(self, epoch):
        ckpt_dir = Path(f"models/checkpoints_tune_{self.opt_g.__class__.__name__}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "G":  self.G.state_dict(),
            "F":  self.F.state_dict(),
            "Dx": self.Dx.state_dict(),
            "Dy": self.Dy.state_dict(),
            "opt_g":  self.opt_g.state_dict(),
            "opt_dx": self.opt_dx.state_dict(),
            "opt_dy": self.opt_dy.state_dict(),
        }, ckpt_dir/f"ckpt_{epoch}.pth")

    def run(self):
        for epoch in range(1, self.epochs+1):
            tr_g, tr_dx, tr_dy = self.train_epoch()
            val_g, val_dx, val_dy = self.validate()

            self.logger.add_scalar("Loss_train/train_g", tr_g, epoch)
            self.logger.add_scalar("Loss_train/train_dx", tr_dx, epoch)
            self.logger.add_scalar("Loss_train/train_dy", tr_dy, epoch)
            self.logger.add_scalar("Loss_val/val_g",   val_g, epoch)
            self.logger.add_scalar("Loss_val/val_dx",  val_dx, epoch)
            self.logger.add_scalar("Loss_val/val_dy",  val_dy, epoch)

            print(
                f"[{epoch}/{self.epochs}] "
                f"Train G:{tr_g:.4f} Dx:{tr_dx:.4f} Dy:{tr_dy:.4f} | "
                f" Val  G:{val_g:.4f} Dx:{val_dx:.4f} Dy:{val_dy:.4f}"
            )

            if epoch % self.save_every == 0:
                self.save_ckpt(epoch)
        self.save_ckpt(self.epochs)
        print("Training complete!")
        self.logger.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer_g",    choices=OPTIMIZERS, required=True)
    p.add_argument("--optimizer_d",    choices=OPTIMIZERS, required=True)
    p.add_argument("--he-dir",         default="data/dataset_HE/tiles")
    p.add_argument("--ki-dir",         default="data/dataset_Ki67/tiles")
    p.add_argument("--he-filt",        default="data/HE_filtered")
    p.add_argument("--ki-filt",        default="data/Ki67_filtered")
    p.add_argument("--batch-size",     type=int,   default=2)
    p.add_argument("--lr-g",           type=float, default=2e-4)
    p.add_argument("--lr-d",           type=float, default=1e-5)
    p.add_argument("--weight-decay",   type=float, default=0.0)
    p.add_argument("--lambda-cycle",   type=float, default=10.0)
    p.add_argument("--lambda-id",      type=float, default=0.5)
    p.add_argument("--grad-clip",      type=float, default=None)
    p.add_argument("--disc-steps",     type=int,   default=1)
    p.add_argument("--label-smoothing",type=float, default=0.0)
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--resume",         type=str,   default=None,
                   help="Путь к .pth для дообучения")
    p.add_argument("--save-every",     type=int,   default=10)
    p.add_argument("--logdir",         default="runs")
    args = p.parse_args()

    trainer = CycleGANTrainer(args)
    trainer.run()

    out_dir = Path(f"models/run_tune_{trainer.opt_g.__class__.__name__}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.G.state_dict(), out_dir / "G_XtoY.pt")
    torch.save(trainer.F.state_dict(), out_dir / "F_YtoX.pt")
    torch.save(trainer.Dx.state_dict(),    out_dir / "D_X.pt")
    torch.save(trainer.Dy.state_dict(),    out_dir / "D_Y.pt")