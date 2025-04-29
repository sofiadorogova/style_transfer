import math
import torch

class OGDA(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(OGDA, self).__init__(params, defaults)
        # Инициализируем словарь для хранения предыдущих градиентов
        self.prev_grads = {id(p): torch.zeros_like(p) for group in self.param_groups for p in group['params']}

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                prev_grad = self.prev_grads[id(p)]
                # Оптимистическое обновление: p = p - lr * (2 * grad - prev_grad)
                p.data.add_(-group['lr'] * (2 * grad - prev_grad))
                # Сохраняем текущий градиент как предыдущий
                self.prev_grads[id(p)] = grad.clone()
                

class OptimisticAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(OptimisticAdam, self).__init__(params, defaults)
        # Инициализируем состояние для каждого параметра
        self.state = {id(p): {
            'step': 0,
            'exp_avg': torch.zeros_like(p),
            'exp_avg_sq': torch.zeros_like(p),
            'prev_grad': torch.zeros_like(p)
        } for group in self.param_groups for p in group['params']}

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[id(p)]
                state['step'] += 1
                beta1, beta2 = group['betas']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                prev_grad = state['prev_grad']

                # Оптимистический градиент: 2 * grad - prev_grad
                optimistic_grad = 2 * grad - prev_grad

                # Обновляем экспоненциальные скользящие средние
                exp_avg.mul_(beta1).add_(optimistic_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(optimistic_grad, optimistic_grad, value=1 - beta2)

                # Коррекция смещения
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])

                # Обновляем параметры
                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Сохраняем текущий градиент
                state['prev_grad'] = grad.clone()
                
