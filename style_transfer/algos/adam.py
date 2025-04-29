import torch

class Adam:
    def __init__(self, params, lr, weight_decay=0.0, maximize=False):
        self.opt = torch.optim.Adam(
            params, lr=lr, weight_decay=weight_decay, betas=(0.5, 0.999)
        )
        self.maximize = maximize

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self, closure):
        # closure() должен вернуть тензор loss
        loss = closure()
        # для максимизации можно инвертировать градиенты:
        if self.maximize:
            for p in self.opt.param_groups[0]['params']:
                if p.grad is not None:
                    p.grad.mul_(-1)
        self.opt.step()
        return loss

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, sd):
        self.opt.load_state_dict(sd)