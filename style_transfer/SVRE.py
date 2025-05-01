import torch
import math
import random

#Реализация самого оптимизатора
class SVRE(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, p=0.1):
        defaults = dict(lr=lr)
        super(SVRE, self).__init__(params, defaults)
        self.p = p
        self.epoch_length = self._sample_epoch_length()
        self.snapshot_params = [p.clone().detach() for group in self.param_groups for p in group['params']]
        self.snapshot_grads = [torch.zeros_like(p) for p in self.snapshot_params]
        self.mu = [torch.zeros_like(p) for p in self.snapshot_params]

    def _sample_epoch_length(self):
        # Имитация геометрического распределения
        return int(math.ceil(math.log(1.0 - random.random()) / math.log(1.0 - self.p)))

    def set_snapshot(self):
        self.snapshot_params = [p.clone().detach() for group in self.param_groups for p in group['params']]

    def set_mu(self, mu_values):
        self.mu = mu_values

    def set_snapshot_grads(self, grads):
        self.snapshot_grads = grads

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                snapshot_grad = self.snapshot_grads[idx]
                mu = self.mu[idx]

                # SVRE шаг: используем variance-reduced градиент
                adjusted_grad = grad - snapshot_grad + mu
                p.data.add_(-group['lr'] * adjusted_grad)

                idx += 1

        return loss

#Функции, которые нужны для правильной работы оптимизатора:

def compute_full_gradient(model, loader, loss_fn):
    model.zero_grad()
    full_grads = [torch.zeros_like(p) for p in model.parameters()]
    count = 0

    for x, y in loader:
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()

        for i, p in enumerate(model.parameters()):
            full_grads[i] += p.grad.detach()
        model.zero_grad()
        count += 1

    return [g / count for g in full_grads]

# Вычисление градиента в snapshot-параметрах:
def compute_snapshot_grads(model, snapshot_params, x, y, loss_fn):
    original_params = [p.clone() for p in model.parameters()]

    # Заменим параметры модели на snapshot
    with torch.no_grad():
        for p, snap in zip(model.parameters(), snapshot_params):
            p.copy_(snap)

    out = model(x)
    loss = loss_fn(out, y)
    model.zero_grad()
    loss.backward()

    grads = [p.grad.detach().clone() for p in model.parameters()]

    # Вернуть параметры обратно
    with torch.no_grad():
        for p, orig in zip(model.parameters(), original_params):
            p.copy_(orig)

    return grads

#Далее идет пример использования

loader = # твой loader

# Модель, лосс, оптимизатор
model = # твоя модель
loss_fn = # твоя функция потерь
optimizer = SVRE(model.parameters(), lr=1e-2, p=0.1)

# Обучение
for epoch in range(5):
    print(f"Epoch {epoch + 1}")

    # Обновим snapshot и полный градиент
    optimizer.set_snapshot()
    mu = compute_full_gradient(model, loader, loss_fn)
    optimizer.set_mu(mu)

    for x_batch, y_batch in loader:
        # Forward на текущем батче
        out = model(x_batch)
        loss = loss_fn(out, y_batch)
        optimizer.zero_grad()
        loss.backward()

        # Вычислить snapshot градиенты на том же батче
        snapshot_grads = compute_snapshot_grads(model, optimizer.snapshot_params, x_batch, y_batch, loss_fn)
        optimizer.set_snapshot_grads(snapshot_grads)

        # Шаг оптимизации
        optimizer.step()
