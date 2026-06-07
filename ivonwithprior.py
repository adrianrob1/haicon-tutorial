"""New implementation of IVON using a general prior."""

from contextlib import contextmanager

import torch
from torch import Tensor

from torch.optim.optimizer import (
    _get_scalar_dtype,
    _get_value,
    Optimizer,
    ParamsT,
)


__all__ = ['IVON', 'ivon']


class IVON(Optimizer):

    def __init__(
        self,
        params: ParamsT,
        ess: float,
        hess_init: float,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.9999),
        weight_decay: float = 1e-6,
        *,
        mc_samples: int = 1,
        clip_radius: float = float('inf'),
        sync: bool = False,
    ) -> None:
        if not 0.0 < ess:
            raise ValueError(f'Invalid effective sampling size: {ess}')
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if not 1 <= mc_samples:
            raise ValueError(f'Invalid mc_samples value: {mc_samples}')
        if not 0 < clip_radius:
            raise ValueError(f'Invalid clip_radius value: {clip_radius}')

        defaults = {
            'ess': ess,
            'hess_init': hess_init,
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'clip_radius': clip_radius,
        }
        super().__init__(params, defaults)

        # set prior precision based on weight-decay
        for group in self.param_groups:
            group['prior_precision'] = group['ess'] * group['weight_decay']

        self._mc_samples = mc_samples
        self._sync = sync

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for p in group['params']:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state['step']):
                    step_val = float(p_state['step'])
                    p_state['step'] = torch.tensor(step_val, dtype=_get_scalar_dtype())

    def _params(self):
        return [p for g in self.param_groups for p in g['params']]

    def _iter_group_params(self):
        for g in self.param_groups:
            for p in g['params']:
                yield g, p

    def _as_list(self, x, n):
        if x is None or isinstance(x, (float, int)):
            return [x] * n
        if isinstance(x, torch.nn.Module):
            return list(x.parameters())
        return list(x)

    def _to_param(self, x, p):
        if x is None:
            return None
        if isinstance(x, (float, int)):
            return torch.full_like(p, float(x))
        x = x.detach().to(device=p.device, dtype=p.dtype)
        if x.numel() == 1:
            x = x.expand_as(p)
        elif x.shape != p.shape:
            raise ValueError(f'expected shape {tuple(p.shape)}, got {tuple(x.shape)}')

        return x.clone()

    def _hessian(self, group, p):
        return self.state[p].get('hessian', torch.full_like(p, group['hess_init']))

    def _prior_mean(self, p):
        return self.state[p].get('prior_mean', None)

    def _prior_precision(self, group, p):
        return self.state[p].get('prior_precision', group['prior_precision'])

    def _posterior_precision(self, group, p):
        return group['ess'] * self._hessian(group, p) + self._prior_precision(group, p)

    @property
    def prior(self):
        means, precisions = [], []
        for g, p in self._iter_group_params():
            means.append(self._prior_mean(p))
            precisions.append(self._prior_precision(g, p))

        return means, precisions

    @prior.setter
    @torch.no_grad()
    def prior(self, value):
        mean, precision = value
        params = self._params()

        means = self._as_list(mean, len(params))
        precisions = self._as_list(precision, len(params))

        if len(means) != len(params) or len(precisions) != len(params):
            raise ValueError('prior mean/precision must match optimizer parameters')

        if isinstance(precision, (float, int)):
            for g in self.param_groups:
                g['prior_precision'] = float(precision)

            for p in params:
                self.state[p].pop('prior_precision', None)

        for p, m, s in zip(params, means, precisions):
            st = self.state[p]
            m = self._to_param(m, p)
            if m is None:
                st.pop('prior_mean', None)
            else:
                st['prior_mean'] = m

            if not isinstance(precision, (float, int)):
                s = self._to_param(s, p)
                if s is None:
                    st.pop('prior_precision', None)
                else:
                    st['prior_precision'] = s

    @property
    def posterior(self):
        means, precisions = [], []

        for g, p in self._iter_group_params():
            means.append(p.detach())
            precisions.append(self._posterior_precision(g, p).detach())

        return means, precisions

    @posterior.setter
    @torch.no_grad()
    def posterior(self, value):
        mean, precision = value
        params = self._params()

        means = self._as_list(mean, len(params))
        precisions = self._as_list(precision, len(params))

        if len(means) != len(params) or len(precisions) != len(params):
            raise ValueError('prior mean/precision must match optimizer parameters')

        for (g, p), m, q_prec in zip(self._iter_group_params(), means, precisions):
            st = self.state[p]
            m = self._to_param(m, p)
            q_prec = self._to_param(q_prec, p)

            if m is not None:
                p.copy_(m)

            if q_prec is not None:
                s = self._prior_precision(g, p)
                h = (q_prec - s) / g['ess']
                st['hessian'] = h.clone()

    @torch.no_grad()
    def reset_momentum(self):
        for _, p in self._iter_group_params():
            st = self.state[p]
            st['step'] = torch.tensor(0.0, dtype=_get_scalar_dtype())
            st['exp_avg'] = torch.zeros_like(p)

    @torch.no_grad()
    def kl(self):
        total = None

        for g, p in self._iter_group_params():
            m0 = self._prior_mean(p)
            s0 = self._prior_precision(g, p)
            q_prec = self._posterior_precision(g, p)

            diff2 = p.detach().square() if m0 is None else (p.detach() - m0).square()

            u = s0 / q_prec - 1
            local = 0.5 * (u - torch.log1p(u) + s0 * diff2).sum()

            total = local if total is None else total + local

        return total.clamp_min(0)

    def _init_group(
        self, group, params_with_grad, mc_grads, mc_nxgs, exp_avgs, hessians, state_steps, prior_means, prior_precisions
    ):
        """Initialize group and collect parameter states for update step. Synchronizes monte-carlo estimates if
        necessary.

        """
        for p in group['params']:
            state = self.state[p]
            # only add if we have some states for it
            if 'mc_nxg' in state and 'mc_grad' in state:
                params_with_grad.append(p)
                # Lazy state initialization. State contains monte-carlo estimates at this point.
                if "step" not in state:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU
                    # This is because kernel launches are costly on CUDA and XLA.
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)

                if "hessian" not in state:
                    state["hessian"] = torch.full_like(p, group["hess_init"])
                
                exp_avgs.append(state['exp_avg'])
                hessians.append(state['hessian'])
                state_steps.append(state['step'])

                prior_means.append(self._prior_mean(p))
                prior_precisions.append(self._prior_precision(group, p))

                # synchronize if we are distributed
                if self._sync and torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size()
                    for key in ('mc_nxg', 'mc_grad'):
                        torch.distributed.all_reduce(state[key])
                        state[key] /= world_size

                # mc_sample dependent states are reset here
                mc_nxgs.append(state.pop('mc_nxg'))
                mc_grads.append(state.pop('mc_grad'))
                # explicitly remove mc_step from state
                del state['mc_step']

    @contextmanager
    def sampled_params(self, train: bool = True):
        """Context manager to sample a set of parameters.

        Args:
            train (bool, optional): Whether currently training.
        """
        try:
            self._sample_params()
            yield
        finally:
            self._restore_param_average(train)

    def _sample_params(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # sample noise to add to the tensor
                std = self._posterior_precision(group, p).rsqrt()
                noise = torch.randn_like(p).mul_(std)

                # we store the average and noise explicitly
                state['average'] = p.data
                state['noise'] = noise
                # make sure to create a new tensor, so we keep the old one
                p.data = p.data + noise

    def _restore_param_average(self, train: bool = True):
        """Reset parameters to the average.

        Args:
            train (bool, optional): Whether currently training.

        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # pop the average from the state and restore it
                try:
                    p.data = state.pop('average')
                except KeyError:
                    # seems like we hit an error, or never changed the average, skip!
                    continue

                # nothing to do if are not training, or the parameter does not require or have a gradient
                if not train or not p.requires_grad or p.grad is None:
                    noise = state.pop('noise', None)
                    continue

                # compute monte-carlo averages
                if 'mc_step' not in state:
                    # initialize if this is the first monte-carlo step
                    state['mc_step'] = 1
                    # might not need to clone here when mc_samples == 1
                    state['mc_grad'] = p.grad.clone()
                    # price approx., pops noise
                    state['mc_nxg'] = state.pop('noise').mul_(p.grad)
                else:
                    # running average from second monte-carlo step
                    state['mc_step'] += 1
                    # running average of gradient from the second sample onwards
                    state['mc_grad'].lerp_(p.grad, 1.0 / state['mc_step'])
                    # price approx., pops noise
                    state['mc_nxg'].lerp_(state.pop('noise').mul_(p.grad), 1.0 / state['mc_step'])

                # finally, delete the gradient, because we accumulate differently
                p.grad = None

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                for _ in range(self._mc_samples):
                    with self.sampled_params(train=True):
                        loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            mc_grads: list[Tensor] = []
            mc_nxgs: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            hessians: list[Tensor] = []
            state_steps: list[Tensor] = []
            prior_means: list[Tensor] = []
            prior_precisions: list[Tensor] = []
            beta1, beta2 = group['betas']

            # collect necessary parameter states
            # this resets the monte-carlo samples by popping from state
            # also synchronizes monte-carlo estiamtes when necessary
            self._init_group(
                group,
                params_with_grad,
                mc_grads,
                mc_nxgs,
                exp_avgs,
                hessians,
                state_steps,
                prior_means,
                prior_precisions,
            )

            ivon(
                params_with_grad,
                mc_grads,
                mc_nxgs,
                exp_avgs,
                hessians,
                state_steps,
                prior_means,
                prior_precisions,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                ess=group['ess'],
                clip_radius=group['clip_radius'],
            )

        return loss


@torch.no_grad()
def ivon(
    params: list[Tensor],
    mc_grads: list[Tensor],
    mc_nxgs: list[Tensor],
    exp_avgs: list[Tensor],
    hessians: list[Tensor],
    state_steps: list[Tensor],
    prior_means: list[Tensor],
    prior_precisions: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    ess: float,
    clip_radius: float,
) -> None:
    for i, param in enumerate(params):
        grad = mc_grads[i]
        nxg = mc_nxgs[i]
        exp_avg = exp_avgs[i]
        hessian = hessians[i]
        step_t = state_steps[i]
        m0 = prior_means[i]
        h0 = prior_precisions[i] / ess

        # update step
        step_t += 1

        # decay the moment running average gradient
        exp_avg.lerp_(grad, 1 - beta1)

        # might be doable without allocating this memory
        hessian_wd = hessian + h0
        # nxg is the hat-hessian
        hat_hessian = nxg.mul_(hessian_wd).mul_(ess)
        # hessian diff squared, divided by hess + wd
        term3 = (hessian - hat_hessian).pow_(2).div_(hessian_wd)
        # update hessian
        hessian.lerp_(hat_hessian, 1 - beta2).add_(term3, alpha=0.5 * (1 - beta2) ** 2)
        # force posterior precision to be positive
        hessian.clamp_(min=-h0 + 1e-12)

        step = _get_value(step_t)
        bias_correction1 = 1 - beta1**step

        # update hessian_wd
        hessian_wd = torch.add(hessian, h0, out=hessian_wd)
        # compute the step
        torch.div(exp_avg, bias_correction1, out=term3)

        if m0 is None:
            term3.add_(param * h0)
        else:
            term3.add_((param - m0) * h0)

        term3.div_(hessian_wd).clip_(min=-clip_radius, max=clip_radius)

        # update parameters
        param.add_(term3, alpha=-lr)
