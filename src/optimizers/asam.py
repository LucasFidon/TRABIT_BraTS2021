# Based on the official repository for ASAM
# https://github.com/SamsungLabs/ASAM/blob/master/asam.py
# 07/07/21
# Modified the code for ASAM to inherit from torch.optim.Optimizer

import torch

EPSILON = 1.e-16

class ASAM(torch.optim.Optimizer):
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        if not 0.0 <= rho:
            raise ValueError("Invalid value for rho, should be non-negative: %f" % rho)
        if not 0.0 <= eta:
            raise ValueError("Invalid value for eta, should be non-negative: %f" % eta)
        self.model = model
        defaults = dict(rho=rho, eta=eta)
        super(ASAM, self).__init__(model.parameters(), defaults)

        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        rho = self.defaults['rho']
        eta = self.defaults['eta']

        # Compute epsilon directions and weight norms for all gradients
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + EPSILON

        # Compute the adaptive epsilon and apply the ascent step
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(rho / wgrad_norm)
            p.add_(eps)

        self.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        """
        Should be called after the ascent step + a new forward-backward pass
        """
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            # We need to first substract epsilon to go back to
            # where we were before the ascent step
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.zero_grad()

    def zero_grad(self,  set_to_none: bool = False):
        self.optimizer.zero_grad()

    def step(self, closure):
        assert closure is not None, \
            "Sharpness Aware Minimization (SAM) requires closure, but it was not provided." \
            "It implies that mixed-precision (--fp16) is not supported with SAM."
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        self.ascent_step()
        closure()
        self.descent_step()


class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        rho = self.defaults['rho']
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + EPSILON
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()
