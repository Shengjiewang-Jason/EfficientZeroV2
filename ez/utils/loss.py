# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torchrl
import torch.nn as nn
import torch.nn.functional as F
from ez.utils.distribution import SquashedNormal, TruncatedNormal, ContDist
from ez.utils.format import atanh
from ez.utils.format import symlog, symexp, DiscreteSupport
from torch.cuda.amp import autocast as autocast


def cosine_similarity_loss(f1, f2):
    """Cosine Consistency loss function: similarity loss
    Parameters
    """
    f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
    f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
    return -(f1 * f2).sum(dim=1)


def kl_loss(prediction, target):
    return -(torch.log_softmax(prediction, dim=-1) * target).sum(-1)


def symlog_loss(prediction, target):
    return 0.5 * (prediction.squeeze() - symlog(target)) ** 2


def Value_loss(preds, targets, config):
    v_num = config.train.v_num
    targets = targets.repeat(v_num, 1)
    iql_weight = config.train.IQL_weight
    if not config.train.use_IQL:
        iql_weight = 0.5
    if config.model.value_support.type == 'symlog':
        loss_func = symlog_loss
        reformed_values = symexp(preds).squeeze()
        target_supports = targets
    elif config.model.value_support.type == 'support':
        loss_func = kl_loss
        reformed_values = DiscreteSupport.vector_to_scalar(preds, **config.model.value_support).squeeze()
        target_supports = DiscreteSupport.scalar_to_vector(targets, **config.model.value_support)
    else:
        raise NotImplementedError

    value_error = reformed_values - targets
    value_sign = (value_error > 0).float().detach()
    value_weight = (1 - value_sign) * iql_weight + value_sign * (1 - iql_weight)
    value_loss = (value_weight * loss_func(preds, target_supports)).mean(0)
    return value_loss

def set_requires_grad(net, value):
	"""Enable/disable gradients for a given (sub)network."""
	for param in net.parameters():
		param.requires_grad_(value)



def continuous_loss(policy, target_action, target_policy, target_best_action, mask=None, distribution_type='squashed_gaussian'):
    action_dim = policy.size(1) // 2
    n_branches = target_policy.size(1)
    if distribution_type == 'squashed_gaussian':
        mean, std = policy[:, :action_dim], policy[:, action_dim:]
        distr = SquashedNormal(mean, std)
    elif distribution_type == 'truncated_gaussian':
        mean, std = policy[:, :action_dim], policy[:, action_dim:]
        distr = torchrl.modules.TruncatedNormal(mean, std)
    else:
        raise NotImplementedError

    # full pi loss in Eq. 6 of the paper
    target_action = torch.moveaxis(target_action, 0, 1)
    policy_log_prob = distr.log_prob(target_action).sum(-1)
    policy_log_prob = torch.moveaxis(policy_log_prob, 0, 1)
    fullpi_loss = (-target_policy * policy_log_prob).sum(1)

    # simple pi loss in Eq. 7 of the paper
    target_best_action = target_best_action.clip(-0.999, 0.999)
    if distribution_type != 'truncated_gaussian':
        simplepi_loss = -distr.log_prob(target_best_action).sum(-1)    # simple policy loss of Gumbel MuZero
    else:
        simplepi_loss = -distr.log_prob(target_best_action)

    # choose action loss according to action dim 
    if action_dim == 1:
        loss = fullpi_loss
    else:
        loss = simplepi_loss

    if distribution_type in ['squashed_gaussian', 'truncated_gaussian']:
        ent_action = distr.rsample((1024,))
        ent_action = ent_action.clip(-0.999, 0.999)
        if distribution_type != 'truncated_gaussian':
            ent_log_prob = distr.log_prob(ent_action).sum(-1)
        else:
            ent_log_prob = distr.log_prob(ent_action)
        entropy = -ent_log_prob.mean(0)
    else:
        entropy = distr.entropy().sum(-1)

    if mask is not None:
        loss = loss * mask
        entropy = entropy * mask

    return loss, entropy


class BarlowLoss(nn.Module):
    def __init__(self, lmbda, reduction='mean'):
        super().__init__()
        self.lmbda = lmbda
        self.reduction = reduction

    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        n, d = z1.shape
        # z1 = F.normalize(z1, dim=-1, p=2)
        # z2 = F.normalize(z2, dim=-1, p=2)

        # normalize along batch dim
        z1 = (z1 - z1.mean(0)) / z1.std(0)  # NxD
        z2 = (z2 - z2.mean(0)) / z2.std(0)  # NxD

        # cross correltation matrix
        cor = torch.mm(z1.T, z2)
        cor.div_(n)

        # loss
        on_diag = torch.diagonal(cor).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(cor).pow_(2).sum()

        loss = on_diag + self.lmbda * off_diag

        if self.reduction == 'mean':
            return loss
        else:
            raise ValueError