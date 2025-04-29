import torch

class _OGDA(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 maximize: bool = False):

        defaults = dict(lr=lr,
                        weight_decay=weight_decay,
                        maximize=maximize)
        super().__init__(params, defaults)

        # храним g_{k-1} для каждого тензора
        self.prev_grads = {
            id(p): torch.zeros_like(p.data)
            for group in self.param_groups
            for p in group['params']
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            maximize = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if maximize:              # превращаем &laquo;максимизировать&raquo; в &laquo;минимизировать&raquo;
                    grad = -grad
                if wd != 0.0:             # L2 regularisation
                    grad = grad.add(p.data, alpha=wd)

                prev_grad = self.prev_grads[id(p)]
                # θ &larr; θ − lr * (2·g − g_{prev})
                p.add_(2 * grad - prev_grad, alpha=-lr)
                self.prev_grads[id(p)] = grad.clone()

        return loss


class OGDA:
    def __init__(self, params, lr, weight_decay=0.0, maximize=False):
        # оригинальный OGDA уже умеет maximize=True/False
        self.opt = _OGDA(params, lr=lr, weight_decay=weight_decay, maximize=maximize)

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self, closure):
        # OGDA.step(closure) возвращает loss
        return self.opt.step(closure)

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, sd):
        self.opt.load_state_dict(sd)