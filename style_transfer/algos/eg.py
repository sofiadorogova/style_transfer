import torch
from torch.optim import Optimizer


class EG(Optimizer):
    """
    Extra‑Gradient
    Параметры группы:
    • lr (float) – шаг обучения;
    • weight_decay (float) – L2‑регуляризация;
    • maximize (bool) – True ⇢ градиентный ПОДЪЁМ (исп. для дискриминатора),
                         False ⇢ градиентный СПУСК (генератор).

    Замечание. Аргумент beta сохранён для совместимости,
    но в этой реализации не используется (momentum = 0).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta: float = 0.0,           # not used, но оставлен для API‑совместимости
        weight_decay: float = 0.0,
        maximize: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)

        # Буфер предсказательного шага для каждого тензора
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["lookahead"] = torch.zeros_like(p.data)

    # ------------------------------------------------------------------ #
    # Стандартные методы Optimizer
    # ------------------------------------------------------------------ #
    def zero_grad(self, set_to_none: bool = False):
        """Полное обнуление градиентов."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()

    def step(self, closure=None):
        """
        Одна итерация Extra‑Gradient: (1) look‑ahead, (2) пересчёт &nabla;,
        (3) финальный апдейт. closure обязан выполнять:
            optimizer.zero_grad(); loss.backward(); return loss
        """
        if closure is None:
            raise RuntimeError("EG requires a closure that reevaluates the model")

        # --- Pass 0: градиент в x_k уже посчитан к моменту вызова step() ---
        with torch.enable_grad():
            loss = closure()  # (backward‑1)

        # ------------------------------------------------------------------ #
        # Pass 1: строим look‑ahead z = x_k &plusmn; lr∙&nabla;f(x_k)
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            for group in self.param_groups:
                lr, wd, maximize = group["lr"], group["weight_decay"], group["maximize"]
                sign = 1.0 if maximize else -1.0

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if wd != 0:
                        p.grad.add_(p.data, alpha=wd)
                    state = self.state[p]
                    state["previous_param"] = p.data.clone()         # x_k
                    state["lookahead"].copy_(p.data)
                    state["lookahead"].add_(p.grad, alpha=sign * lr)  # z

        # ------------------------------------------------------------------ #
        # Pass 2: градиент в точке z = x_{k+½}
        # ------------------------------------------------------------------ #
        # Временно подменяем веса look‑ahead‑значениями
        orig_data = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                orig_data.append((p, p.data.clone()))
                p.data.copy_(self.state[p]["lookahead"])

        with torch.enable_grad():
            loss = closure()  # backward‑2

        # Возвращаем x_k, но оставляем p.grad = &nabla;f(z) ---------------------- #
        for p, old in orig_data:
            p.data.copy_(old)

        # ------------------------------------------------------------------ #
        # Pass 3: финальное обновление x_{k+1} = x_k &plusmn; lr∙&nabla;f(z)
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            for group in self.param_groups:
                lr, wd, maximize = group["lr"], group["weight_decay"], group["maximize"]
                sign = 1.0 if maximize else -1.0

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if wd != 0:
                        p.grad.add_(p.data, alpha=wd)

                    p.data.copy_(self.state[p]["previous_param"])  # &larr; x_k
                    p.data.add_(p.grad, alpha=sign * lr)           # &larr; x_{k+1}

        return loss