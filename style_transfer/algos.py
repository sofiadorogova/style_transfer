import torch
from torch.optim import Optimizer


class EG(Optimizer):
    """Реализация метода Extra Gradient"""

    def __init__(self, params, lr=1e-3, beta=0.9, weight_decay=0):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super(EG, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['lookahead'] = torch.zeros_like(p.data)

    def zero_grad(self):
        """Обнуляет градиенты оптимизированных параметров"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Compute (x_{k+1/2}, y_{k+1/2})
        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                weight_decay = group['weight_decay']

                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    if weight_decay != 0:
                        p.grad.add_(p.data, alpha=weight_decay)

                    state['previous_param'] = p.data.clone()

                    # x_{k+1/2} = x_k - η∇f(x_k)
                    # y_{k+1/2} = y_k + η∇f(y_k)
                    sign = -1 if i % 2 == 0 else 1
                    state['lookahead'].copy_(p.data)
                    state['lookahead'].add_(p.grad, alpha=sign*lr)

        # Second pass: compute gradients at extrapolated point
        with torch.enable_grad():
            # Temporarily swap parameters with lookahead values
            original_params = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    original_params.append((p, p.data.clone()))
                    p.data.copy_(state['lookahead'])

            # Recompute loss and gradients at (x_{k+1/2}, y_{k+1/2})
            if closure is not None:
                loss = closure()

            # Restore original parameters but keep new gradients
            for p, original_data in original_params:
                p.data.copy_(original_data)

        # Final update using gradients at extrapolated point
        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                weight_decay = group['weight_decay']

                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    if weight_decay != 0:
                        p.grad.add_(p.data, alpha=weight_decay)

                    # x_{k+1} = x_k - η∇f(x_{k+1/2})
                    # y_{k+1} = y_k + η∇f(y_{k+1/2})
                    sign = -1 if i % 2 == 0 else 1
                    p.data.copy_(state['previous_param'])
                    p.data.add_(p.grad, alpha=sign*lr)

        return loss
