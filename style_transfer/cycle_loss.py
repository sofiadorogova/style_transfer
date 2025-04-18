import torch
import torch.nn as nn
import torch.nn.functional as F

class CycleGANLoss(nn.Module):
    """
    Лосс для CycleGAN.
    Считает:
      - Adversarial loss для G (обмануть D_Y) и F (обмануть D_X)
      - Cycle-consistency loss: x->G->F->x, y->F->G->y
      - Identity Loss
    """
    def __init__(self, lambda_cycle: float = 10.0, lambda_id: float = 0.5):
        """
        :param lambda_cycle: вес cycle-consistency лосса (в статье берут 10.0)
        :param lambda_id: вес identity loss
        """
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self,
                # Изображения из домена X и Y
                real_x: torch.Tensor, 
                real_y: torch.Tensor,
                # Фейковые (G(X), F(Y))
                fake_y: torch.Tensor, 
                fake_x: torch.Tensor,
                # Реконструкции (F(G(X)), G(F(Y)))
                rec_x: torch.Tensor, 
                rec_y: torch.Tensor,
                # Выходы дискриминаторов на фейках
                d_y_fake: torch.Tensor,  # = D_Y(fake_y)
                d_x_fake: torch.Tensor   # = D_X(fake_x)
                ) -> torch.Tensor:
        """
        Возвращает суммарный лосс генераторов G и F.

        Пояснение аргументов:
          - real_x: batch из домена X
          - real_y: batch из домена Y
          - fake_y = G(real_x)
          - fake_x = F(real_y)
          - rec_x = F(fake_y) = F(G(real_x))
          - rec_y = G(fake_x) = G(F(real_y))
          - d_y_fake = D_Y(fake_y)
          - d_x_fake = D_X(fake_x)
        """

        #Adversarial loss для G и F (LSGAN)
        adv_g = self.mse(d_y_fake, torch.ones_like(d_y_fake)) # G хочет, чтобы D_Y(fake_y) ~ 1
        adv_f = self.mse(d_x_fake, torch.ones_like(d_x_fake)) # F хочет, чтобы D_X(fake_x) ~ 1

        # Cycle-consistency (L1)
        cycle_x = self.l1(rec_x, real_x)  # F(G(x)) ~ x
        cycle_y = self.l1(rec_y, real_y)  # G(F(y)) ~ y
        cycle_loss = cycle_x + cycle_y

        # Identity loss
        id_x = self.l1(fake_y - real_x)   # G(x) - x
        id_y = self.l1(fake_x - real_y)   # F(y) - y
        id_loss = id_x + id_y

        loss_g = adv_g + adv_f + self.lambda_cycle * cycle_loss + self.lambda_id * id_loss
        return loss_g