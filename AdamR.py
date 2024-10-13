import math
import torch
from torch.optim.optimizer import Optimizer
import copy

class AdamR(Optimizer):
    r"""Implements Adam with stable delta parameter decay (AdamR) algorithm.
    It has be proposed in 
    `Stable Weight Decay Regularization`__.
    Follow AdamS: Scheduled (Stable) Weight Decay

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-4)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(self, params,pre_model, lr=1e-3, betas=(0.9, 0.999), L1=False,eps=1e-8,
                 weight_decay=1e-4, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(pre_model = pre_model,lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,L1=L1)
        super(AdamR, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamR, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size = 0
        exp_avg_sq_hat_sum = 0.
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_size += p.numel()

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamR does not support sparse gradients')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                beta1, beta2 = group['betas']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    exp_avg_sq_hat = max_exp_avg_sq / bias_correction2
                else:
                    exp_avg_sq_hat = exp_avg_sq / bias_correction2
                
                exp_avg_sq_hat_sum += exp_avg_sq_hat.sum()

        # Calculate the sqrt of the mean of all elements in exp_avg_sq_hat  
        exp_avg_mean_sqrt = math.sqrt(exp_avg_sq_hat_sum / param_size)
        for group in self.param_groups:
            weight_update = copy.deepcopy(group['pre_model'])
            for p, preweight in zip(group['params'], weight_update):
                preweight = preweight.to(p.device)
                if p.grad is None:
                    continue

                state = self.state[p]

                #Perform stable weight decay
                if group['weight_decay'] !=0:
                    if exp_avg_mean_sqrt == 0:
                        exp_avg_mean_sqrt =1
                    if group['L1']:
                        p.data.add_(-group['weight_decay'] * group['lr'] / exp_avg_mean_sqrt * (torch.sign(p.data - preweight.data)))
                    else:
                        p.data.add_(-group['weight_decay'] * group['lr'] / exp_avg_mean_sqrt * (p.data - preweight.data))
                        print('####### activate regularizer')
                    preweight = preweight.cpu()
                beta1, beta2 = group['betas']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    exp_avg_sq_hat = max_exp_avg_sq / bias_correction2
                else:
                    exp_avg_sq_hat = exp_avg_sq / bias_correction2
                
                denom = exp_avg_sq_hat.sqrt().add(group['eps'])
                
                step_size = group['lr'] / bias_correction1 
                p.addcdiv_(exp_avg, denom, value= - step_size)
                
        return loss