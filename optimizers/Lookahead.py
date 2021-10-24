from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer




class Lookahead(Optimizer):
    r"""PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups

        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]

        self.state = defaultdict(dict)
        self.state['_la_step'] = la_alpha
        self.state['la_alpha'] = la_alpha
        self.state['_total_la_steps'] = la_steps
        self.state['pullback_momentum'] = pullback_momentum

        # Cache the current optimizer parameters
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.state['pullback_momentum'] == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)


    # def __getstate__(self):
    #     return {
    #         'state': self.state,
    #         'optimizer': self.optimizer,
    #         'la_alpha': self.la_alpha,
    #         '_la_step': self._la_step,
    #         '_total_la_steps': self._total_la_steps,
    #         'pullback_momentum': self.pullback_momentum
    #     }

    def __repr__(self):
        return f"Lookahead()"

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self.state['_la_step']

    def state_dict(self):
        inner_opt_state = self.optimizer.state_dict()['state']
        # LA_state = {(id(k) if isinstance(k, torch.Tensor) else k): v for k, v in self.state.items()}
        LA_state = super(Lookahead, self).state_dict()['state']
        param_groups = self.optimizer.state_dict()['param_groups']
        return {
            "inner_opt_state": inner_opt_state,
            "LA_state": LA_state,
            "param_groups": param_groups
        }

    def load_state_dict(self, state_dict):
        inner_opt_state = {
            "state": state_dict['inner_opt_state'],
            "param_groups": state_dict['param_groups']
        }
        self.optimizer.load_state_dict(inner_opt_state)

        LA_state = {
            "state": state_dict['LA_state'],
            "param_groups": state_dict['param_groups']
        }
        super(Lookahead, self).load_state_dict(LA_state)

        self.param_groups = self.optimizer.param_groups

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']


    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self.state['_la_step'] += 1

        if self.state['_la_step'] >= self.state['_total_la_steps']:
            self.state['_la_step'] = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.state['la_alpha']).add_(param_state['cached_params'], alpha=1.0 - self.state['la_alpha'])  # crucial line
                    param_state['cached_params'].copy_(p.data)

                    if self.state['pullback_momentum'] == "pullback":
                        self.optimizer.state[p]["momentum_buffer"].mul_(self.state['la_alpha']).add_(
                            param_state["cached_mom"], alpha=1.0 - self.state['la_alpha'])
                        param_state["cached_mom"].copy_(self.optimizer.state[p]["momentum_buffer"])
                    elif self.state['pullback_momentum'] == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss


# optimizer = # {any optimizer} e.g. torch.optim.Adam
# if args.lookahead:
#     optimizer = Lookahead(optimizer, la_steps=args.la_steps, la_alpha=args.la_alpha)