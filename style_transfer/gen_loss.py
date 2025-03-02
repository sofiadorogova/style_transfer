import torch
import torch.nn as nn
import torch.nn.functional as F


class GenLoss(nn.Module):
    """
    Лосс-функция для генератора из статьи.
    """
    def __init__(self, alpha: float = 0.02, beta: float = 2000.):
        """
        Сохраняет параметры alpha и beta -- веса в лоссе.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, fake_image: torch.Tensor, real_image: torch.Tensor,
                dis_pred_fake: torch.Tensor) -> torch.Tensor:
        """
        Последовательно вычисляет L1-лосс, полную дисперсию и генеративный лосс,
        возвращает взвешенную сумму.
        """
        l1_loss = F.l1_loss(fake_image, real_image)
        tv_loss = (fake_image ** 2).mean(dim=0).sum()
        gen_loss = ((1. - dis_pred_fake) ** 2).mean()
        loss = l1_loss + self.alpha * tv_loss + self.beta * gen_loss
        return loss
