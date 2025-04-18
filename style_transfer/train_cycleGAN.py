from pathlib import Path
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json

import torch.nn.utils as utils
from cycle_loss import CycleGANLoss
from networks import UNet, VGGDiscriminator
from dataset import StainDataset

class CycleGANTrainer:
    def __init__(self,
                 G_XtoY, F_YtoX,
                 D_X, D_Y,
                 dataset,
                 batch_size=8,
                 lr_g=1e-4,
                 lr_d=1e-5,
                 lambda_cycle=10.0,
                 lambda_id = 0.5,
                 epochs=50,
                 save_every=10,
                 grad_clip_value=1.0):

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.save_every = save_every
        self.grad_clip_value = grad_clip_value
        self.lambda_id = lambda_id

        # Модели
        self.G_XtoY = G_XtoY.to(self.device)
        self.F_YtoX = F_YtoX.to(self.device)
        self.D_X = D_X.to(self.device)
        self.D_Y = D_Y.to(self.device)

        # Разделяем датасет на train/val
        number_generator = torch.Generator().manual_seed(42)
        train_data, val_data, test_data = random_split(dataset, [0.6, 0.2, 0.2], number_generator)

        train_indices = train_data.indices
        val_indices   = val_data.indices
        test_indices  = test_data.indices

        split_dict = {
            "train": list(train_indices),
            "val":   list(val_indices),
            "test":  list(test_indices)
        }

        with open("split_indices.json", "w") as f:
            json.dump(split_dict, f)

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        # Оптимизаторы (один для G + F, отдельные для D_X и D_Y)
        self.opt_g = torch.optim.Adam(
            list(self.G_XtoY.parameters()) + list(self.F_YtoX.parameters()),
            lr=lr_g, betas=(0.5, 0.999)
        )
        self.logger = SummaryWriter()
        self.opt_dx = torch.optim.Adam(self.D_X.parameters(), lr=lr_d, betas=(0.5, 0.999))
        self.opt_dy = torch.optim.Adam(self.D_Y.parameters(), lr=lr_d, betas=(0.5, 0.999))

        # Лосс: CycleGANLoss (из cycle_loss.py)
        self.cycle_loss_fn = CycleGANLoss(lambda_cycle=lambda_cycle)
        # MSE для дискриминаторов (LSGAN)
        self.mse = nn.MSELoss()
    
    def train_epoch(self):
        """Одна эпоха обучения"""
        self.G_XtoY.train()
        self.F_YtoX.train()
        self.D_X.train()
        self.D_Y.train()

        total_g_loss = 0.0
        total_dx_loss = 0.0
        total_dy_loss = 0.0
        steps = 0

        for real_x, real_y in tqdm(self.train_loader, desc="Train step"):
            real_x, real_y = real_x.to(self.device), real_y.to(self.device)

            self.opt_dx.zero_grad() #занулили градиенты

            # 1) Генерируем
            fake_y = self.G_XtoY(real_x)
            rec_x  = self.F_YtoX(fake_y)

            fake_x = self.F_YtoX(real_y)
            rec_y  = self.G_XtoY(fake_x)

            # 2) Обновляем D_x

            # Генерация фейка
            dx_real = self.D_X(real_x) 
            dx_loss_real = self.mse(dx_real, torch.ones_like(dx_real))

            dx_fake = self.D_X(fake_x.detach())
            dx_loss_fake = self.mse(dx_fake, torch.zeros_like(dx_fake))

            dx_loss = 0.5 * (dx_loss_real + dx_loss_fake)
            dx_loss.backward()

            # --- клиппим градиенты ---
            utils.clip_grad_norm_(self.D_X.parameters(), max_norm=self.grad_clip_value)

            self.opt_dx.step()

            # 3) Обновляем D_Y
            self.opt_dy.zero_grad()
            dy_real = self.D_Y(real_y)
            dy_loss_real = self.mse(dy_real, torch.ones_like(dy_real))

            dy_fake = self.D_Y(fake_y.detach())
            dy_loss_fake = self.mse(dy_fake, torch.zeros_like(dy_fake))

            dy_loss = 0.5 * (dy_loss_real + dy_loss_fake)
            dy_loss.backward()

            # --- клиппим градиенты ---
            utils.clip_grad_norm_(self.D_Y.parameters(), max_norm=self.grad_clip_value)

            self.opt_dy.step()

            # 4) Обновляем генераторы (G и F)
            self.opt_g.zero_grad()

            # Повторяем генерацию (чтобы было без detach, иначе не передадутся градиенты)
            d_y_fake_for_g = self.D_Y(fake_y)
            d_x_fake_for_f = self.D_X(fake_x)

            g_loss = self.cycle_loss_fn(
                real_x, real_y,
                fake_y, fake_x,
                rec_x, rec_y,
                d_y_fake_for_g, d_x_fake_for_f
            )

            # Вычисление IdentityLoss (только в train)
            same_y = self.G_XtoY(real_y)
            loss_identity_y = self.L1_loss(same_y, real_y)
            same_x = self.F_YtoX(real_x)
            loss_identity_x = self.L1_loss(same_x, real_x)
            identity_loss = self.lambda_id * (loss_identity_x + loss_identity_y)

            g_loss += identity_loss

            g_loss.backward()

            # --- клиппим градиенты ---
            utils.clip_grad_norm_(
                list(self.G_XtoY.parameters()) + list(self.F_YtoX.parameters()),
                max_norm=self.grad_clip_value
            )

            self.opt_g.step()

            # Сохраняем лоссы
            total_g_loss  += g_loss.item()
            total_dx_loss += dx_loss.item()
            total_dy_loss += dy_loss.item()
            steps += 1

        return (total_g_loss/steps, total_dx_loss/steps, total_dy_loss/steps)

    def validate(self):
        """Оценка на валидационном датасете"""
        self.G_XtoY.eval()
        self.F_YtoX.eval()
        self.D_X.eval()
        self.D_Y.eval()

        val_g_loss  = 0.0
        val_dx_loss = 0.0
        val_dy_loss = 0.0
        steps = 0

        with torch.no_grad():
            for real_x, real_y in tqdm(self.val_loader, desc="Val step"):
                real_x, real_y = real_x.to(self.device), real_y.to(self.device)

                # Генерация
                fake_y = self.G_XtoY(real_x)
                rec_x  = self.F_YtoX(fake_y)
                fake_x = self.F_YtoX(real_y)
                rec_y  = self.G_XtoY(fake_x)
                
                # Подсчёт лоссов D_x
                dx_real = self.D_X(real_x)
                dx_fake = self.D_X(fake_x)
                dx_loss_real = self.mse(dx_real, torch.ones_like(dx_real))
                dx_loss_fake = self.mse(dx_fake, torch.zeros_like(dx_fake))
                dx_loss = 0.5*(dx_loss_real + dx_loss_fake)
                
                # Подсчёт лоссов D_y
                dy_real = self.D_Y(real_y)
                dy_fake = self.D_Y(fake_y)
                dy_loss_real = self.mse(dy_real, torch.ones_like(dy_real))
                dy_loss_fake = self.mse(dy_fake, torch.zeros_like(dy_fake))
                dy_loss = 0.5*(dy_loss_real + dy_loss_fake)

                # Лосс генераторов
                d_y_fake_for_g = self.D_Y(fake_y)
                d_x_fake_for_f = self.D_X(fake_x)
                g_loss = self.cycle_loss_fn(
                    real_x, real_y,
                    fake_y, fake_x,
                    rec_x, rec_y,
                    d_y_fake_for_g, d_x_fake_for_f
                )

                val_g_loss  += g_loss.item()
                val_dx_loss += dx_loss.item()
                val_dy_loss += dy_loss.item()
                steps += 1

        return (val_g_loss/steps, val_dx_loss/steps, val_dy_loss/steps)

    def save_checkpoint(self, epoch):
        """
        Сохраняем checkpoint (модели + оптимизаторы), чтобы можно было продолжить обучение.
        """
        checkpoint_dir = "models/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "G_XtoY": self.G_XtoY.state_dict(),
            "F_YtoX": self.F_YtoX.state_dict(),
            "D_X": self.D_X.state_dict(),
            "D_Y": self.D_Y.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_dx": self.opt_dx.state_dict(),
            "opt_dy": self.opt_dy.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Загрузка checkpoint (пример). Можно вызвать до trainer.run().
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Восстанавливаем веса
        self.G_XtoY.load_state_dict(ckpt["G_XtoY"])
        self.F_YtoX.load_state_dict(ckpt["F_YtoX"])
        self.D_X.load_state_dict(ckpt["D_X"])
        self.D_Y.load_state_dict(ckpt["D_Y"])

        # Восстанавливаем оптимизаторы
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_dx.load_state_dict(ckpt["opt_dx"])
        self.opt_dy.load_state_dict(ckpt["opt_dy"])

        start_epoch = ckpt["epoch"] + 1
        print(f"Loaded checkpoint {checkpoint_path}, starting at epoch {start_epoch}")
        return start_epoch

    def run(self):
        for epoch in range(1, self.epochs+1):
            #train
            train_g, train_dx, train_dy = self.train_epoch()
            self.logger.add_scalar("Loss_train/train_g", train_g, epoch)
            self.logger.add_scalar("Loss_train/train_dx", train_dx, epoch)
            self.logger.add_scalar("Loss_train/train_dy", train_dy, epoch)
    
            #val
            val_g, val_dx, val_dy = self.validate()
            self.logger.add_scalar("Loss_val/val_g", val_g, epoch)
            self.logger.add_scalar("Loss_val/val_dx", val_dx, epoch)
            self.logger.add_scalar("Loss_val/val_dy", val_dy, epoch)
            print(f"Epoch [{epoch}/{self.epochs}] | "
                  f"Train G: {train_g:.4f}, DX: {train_dx:.4f}, DY: {train_dy:.4f} | "
                  f"Val G: {val_g:.4f}, DX: {val_dx:.4f}, DY: {val_dy:.4f}")
            
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch)

        print("Training complete!")

if __name__ == "__main__":
    G_XtoY = UNet()           # генератор: HE -> Ki67
    F_YtoX = UNet()           # генератор: Ki67 -> HE
    D_X = VGGDiscriminator()  # дискриминатор для домена X (HE)
    D_Y = VGGDiscriminator()  # дискриминатор для домена Y (Ki67)

    he_dir = Path("data/dataset_HE/tiles")
    ki_dir = Path("data/dataset_Ki67/tiles")
    
    he_filtered_dir = Path("data/HE_filtered")
    ki_filtered_dir = Path("data/Ki67_filtered")

    print("INFO: Загрузка датасета")

    dataset = StainDataset(
        he_dir=he_dir,
        ki_dir=ki_dir,
        he_filtered_dir=he_filtered_dir,
        ki_filtered_dir=ki_filtered_dir,
        save_filtered=True,
        prob_correction=0.5  # 50% вероятность применять баланс белого
    )

    trainer = CycleGANTrainer(
        G_XtoY, F_YtoX, D_X, D_Y,
        dataset=dataset,
        batch_size=8,
        lr_g=1e-4,
        lr_d=1e-5,
        lambda_cycle=10.0,
        epochs=100,
        save_every=10,
        grad_clip_value=1.0
    )
    # Пример (если хотим дообучать с чекпоинта):
    # start_epoch = trainer.load_checkpoint("checkpoint_epoch_50.pt")

    trainer.run()

    torch.save(trainer.G_XtoY.state_dict(), "models/run_3/G_XtoY_final.pt")
    torch.save(trainer.F_YtoX.state_dict(), "models/run_3/F_YtoX_final.pt")
    torch.save(trainer.D_X.state_dict(),    "models/run_3/D_X_final.pt")
    torch.save(trainer.D_Y.state_dict(),    "models/run_3/D_Y_final.pt")
