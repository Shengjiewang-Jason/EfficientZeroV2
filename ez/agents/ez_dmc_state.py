# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import time
import copy
import math
import torch
import torchrl
import torch.nn as nn
from ez.agents.base import Agent
from omegaconf import open_dict

import numpy as np
from ez.envs import make_dmc
from ez.utils.format import DiscreteSupport
from ez.agents.models import EfficientZero

def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ReLU,
    init_zero=False,
    use_bn=True,
    p_norm=False,
    noisy=False
):
    """MLP layers
    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    init_zero: bool
        zero initialization for the last layer (including w and b).
        This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            if use_bn:
                layers += [
                    torchrl.modules.NoisyLinear(sizes[i], sizes[i + 1], std_init=0.5) if noisy else nn.Linear(sizes[i], sizes[i + 1]),
                    nn.BatchNorm1d(sizes[i + 1]),
                    act()
                ]
            else:
                layers += [torchrl.modules.NoisyLinear(sizes[i], sizes[i + 1], std_init=0.5) if noisy else nn.Linear(sizes[i], sizes[i + 1]),
                           act()]
        else:
            if p_norm == True:
                layers += [PNorm()]
            act = output_activation
            layers += [torchrl.modules.NoisyLinear(sizes[i], sizes[i + 1], std_init=0.5) if noisy else nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        if noisy:
            layers[-2].reset_parameters()
        else:
            layers[-2].weight.data.fill_(0)
            layers[-2].bias.data.fill_(0)


    return nn.Sequential(*layers)


# improved residual block from Pre-LN Transformer
# ref: http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf
class ImproveResidualBlock(nn.Module):
    def __init__(self, input_shape, hidden_shape):
        super(ImproveResidualBlock, self).__init__()
        self.ln1 = nn.LayerNorm(input_shape)
        self.linear1 = nn.Linear(input_shape, hidden_shape)
        self.linear2 = nn.Linear(hidden_shape, input_shape)

    def forward(self, x):
        identity = x
        out = self.ln1(x)
        out = self.linear1(out)
        out = nn.functional.relu(out)
        out = self.linear2(out)

        out += identity
        return out


# L2 norm layer on the next to last layer
class PNorm(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        assert len(x.shape) == 2
        return nn.functional.normalize(x, dim=1, eps=self.eps)


class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon=1e-5, momentum=0.1):
        super(RunningMeanStd, self).__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.count = 1e3
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            batch_count = x.shape[0]
            self.running_mean, self.running_var, self.count = self.update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, mean, var, batch_count)
            global_mean = self.running_mean
            global_var = self.running_var
        else:
            global_mean = self.running_mean
            global_var = self.running_var
        x = (x - global_mean) / torch.sqrt(global_var + self.epsilon)
        return x

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        n_stacked_obs,
        num_blocks,
        rep_net_shape,
        hidden_shape,
        use_bn=True,
    ):
        """Representation network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        n_stacked_obs: int
            number of stacked observation
        num_blocks: int
            number of res blocks
        rep_net_shape: int
            shape of hidden layers
        hidden_shape:
            dim of output hidden state
        use_bn: bool
            True -> Batch normalization
        """
        super().__init__()
       
        self.running_mean_std = RunningMeanStd(observation_shape * n_stacked_obs)
        self.mlp = nn.Linear(observation_shape * n_stacked_obs, hidden_shape)
        self.ln = nn.LayerNorm(hidden_shape)
        self.Rep_resblocks = nn.ModuleList(
            [ImproveResidualBlock(hidden_shape, rep_net_shape) for _ in range(num_blocks)]
        )

    def forward(self, x):

        x = self.running_mean_std(x)
        x = self.mlp(x)
        x = self.ln(x)
        x = torch.tanh(x)
        # res block
        for block in self.Rep_resblocks:
            x = block(x)

        return x

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean


# Predict next hidden states given current states and actions
class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        hidden_shape,
        action_shape,
        num_blocks,
        dyn_shape,
        act_embed_shape,
        rew_net_shape,
        reward_support_size,
        init_zero=False,
        use_bn=True,
    ):
        """Dynamics network
        Parameters
        ----------
        hidden_shape: int
            dim of input hidden state
        action_shape: int
            dim of action
        num_blocks: int
            number of res blocks
        dyn_shape: int
            number of nodes of hidden layer
        act_embed_shape: int
            dim of action embedding
        rew_net_shape: list
            hidden layers of the reward prediction head (MLP head)
        reward_support_size: int
            dim of reward output
        init_zero: bool
            True -> zero initialization for the last layer of reward mlp
        use_bn: bool
            True -> Batch normalization
        """
        super().__init__()
        self.hidden_shape = hidden_shape

        self.act_linear1 = nn.Linear(action_shape, act_embed_shape)
        self.act_ln1 = nn.LayerNorm(act_embed_shape)

        self.dyn_ln_1 = nn.LayerNorm(hidden_shape + act_embed_shape)
        self.dyn_net_1 = nn.Linear(hidden_shape + act_embed_shape, dyn_shape)

        self.dyn_ln_2 = nn.LayerNorm(dyn_shape)
        self.dyn_net_2 = nn.Linear(dyn_shape, hidden_shape)

        if num_blocks > 0:
            self.dyn_resblocks = nn.ModuleList(
                [ImproveResidualBlock(hidden_shape, dyn_shape) for _ in range(num_blocks)]
            )
        else:
            self.dyn_resblocks = nn.ModuleList([])


    def forward(self, hidden, action, reward_hidden=None):

        # action embedding
        act_emb = self.act_linear1(action)
        act_emb = self.act_ln1(act_emb)
        act_emb = nn.functional.relu(act_emb)
        # act_emb = nn.functional.tanh(act_emb)

        # imporved res block 1st
        x = self.dyn_ln_1(torch.cat((hidden, act_emb), dim=-1))
        x = self.dyn_net_1(x)
        x = nn.functional.relu(x)
        x = self.dyn_net_2(x)

        state = hidden + x

        # residual tower for dynamic model (2nd -> num blocks)
        for block in self.dyn_resblocks:
            state = block(state)

        # return state
        return state

    def get_dynamic_mean(self):

        mean = []
        for name, param in self.dyn_net_1.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)

        return mean


class RewardNetwork(nn.Module):
    def __init__(
        self,
        hidden_shape,
        rew_net_shape,
        reward_support_size,
        init_zero=False,
        use_bn=True,
    ):
        super().__init__()
        self.hidden_shape = hidden_shape
        self.rew_net_shape = rew_net_shape
        self.reward_support_size = reward_support_size
        self.rew_resblock = ImproveResidualBlock(self.hidden_shape, self.hidden_shape)
        self.ln = nn.LayerNorm(self.hidden_shape)
        self.rew_net = mlp(self.hidden_shape, self.rew_net_shape, self.reward_support_size,
                           init_zero=init_zero,
                           use_bn=use_bn)


    def forward(self, next_state):
        next_state = self.rew_resblock(next_state)
        next_state = self.ln(next_state)
        reward = self.rew_net(next_state)
        return reward


class RewardNetworkLSTM(nn.Module):
    def __init__(
        self,
        hidden_shape,
        rew_net_shape,
        reward_support_size,
        lstm_hidden_size,
        init_zero=False,
        use_bn=True,
    ):
        super().__init__()
        self.hidden_shape = hidden_shape
        self.rew_net_shape = rew_net_shape
        self.reward_support_size = reward_support_size
        self.rew_resblock = ImproveResidualBlock(self.hidden_shape, self.hidden_shape)
        self.ln = nn.LayerNorm(self.hidden_shape)
        self.lstm = nn.LSTM(input_size=self.hidden_shape, hidden_size=lstm_hidden_size)
        self.rew_net = mlp(lstm_hidden_size, self.rew_net_shape, self.reward_support_size,
                           init_zero=init_zero,
                           use_bn=use_bn)

    def forward(self, next_state, hidden):
        next_state = self.rew_resblock(next_state)
        next_state = self.ln(next_state)
        next_state = next_state.unsqueeze(0)
        reward, hidden = self.lstm(next_state, hidden)
        reward = reward.squeeze(0)
        reward = self.rew_net(reward)
        return reward, hidden



# predict the value and policy given hidden states
class ValuePolicyNetwork(nn.Module):
    def __init__(
        self,
        hidden_shape,
        val_net_shape,
        pi_net_shape,
        action_shape,
        full_support_size,
        init_zero=False,
        use_bn=True,
        p_norm=False,
        policy_distr='squashed_gaussian',
        noisy=False,
        value_support=None,
        **kwargs
    ):
        super().__init__()
        self.v_num = kwargs.get('v_num')
        self.hidden_shape = hidden_shape
        self.val_net_shape = val_net_shape
        self.action_shape = action_shape
        self.pi_net_shape = pi_net_shape

        self.action_space_size = action_shape
        self.init_std = 1.0
        self.min_std = 0.1
        self.policy_distr = policy_distr

        self.val_resblock = ImproveResidualBlock(hidden_shape, hidden_shape)
        self.pi_resblock = ImproveResidualBlock(hidden_shape, hidden_shape)

        self.val_ln = nn.LayerNorm(hidden_shape)
        self.pi_ln = nn.LayerNorm(hidden_shape)


        self.val_nets = nn.ModuleList([
            mlp(self.hidden_shape, self.val_net_shape, full_support_size,
                # init_zero=init_zero,
                use_bn=use_bn,
            )
            for _ in range(self.v_num)])
        self.pi_net = mlp(self.hidden_shape, self.pi_net_shape, self.action_shape * 2,
                          init_zero=init_zero,
                          use_bn=use_bn,
                          p_norm=p_norm,
                          noisy=noisy)

        self.noisy = noisy
        self.value_support = value_support

    def reset_noise(self):
        if self.noisy:
            for layer in self.pi_net:
                try:
                    layer.reset_noise()
                except:
                    pass

    def forward(self, x):
        value = self.val_resblock(x)
        value = self.val_ln(value)
        values = []
        for val_net in self.val_nets:
            values.append(val_net(value))
        values = torch.stack(values)

        policy = self.pi_resblock(x)
        policy = self.pi_ln(policy)
        policy = self.pi_net(policy)

        action_space_size = policy.shape[-1] // 2
        if self.policy_distr == 'squashed_gaussian':
            policy[:, :action_space_size] = 5 * torch.tanh(policy[:, :action_space_size] / 5)  # soft clamp mu
            policy[:, action_space_size:] = torch.nn.functional.softplus(
                policy[:, action_space_size:] + self.init_std) + self.min_std  # same as Dreamer-v3

        return values, policy

    def log_std(self, x, low, dif):
        return low + 0.5 * dif * (torch.tanh(x) + 1)


class EZDMCStateAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

        self.update_config()

        self.num_blocks = config.model.num_blocks
        self.fc_layers = config.model.fc_layers
        self.down_sample = config.model.down_sample
        self.state_norm = config.model.state_norm
        self.value_prefix = config.model.value_prefix
        self.init_zero = config.model.init_zero
        self.value_ensumble = config.model.value_ensumble
        self.value_policy_detach = config.train.value_policy_detach
        self.v_num = config.train.v_num

    def update_config(self):
        assert not self._update
        env = make_dmc(self.config.env.game, seed=0, save_path=None, **self.config.env)
        obs = env.reset()
        obs_shape = obs.shape[0]
        action_space_size = env.action_space.shape[0]

        reward_support = DiscreteSupport(self.config)
        reward_size = reward_support.size
        self.reward_support = reward_support

        value_support = DiscreteSupport(self.config)
        value_size = value_support.size
        self.value_support = value_support

        localtime = time.strftime('%Y-%m-%d %H:%M:%S')
        tag = '{}-seed={}-{}/'.format(self.config.tag, self.config.env.base_seed, localtime)

        with open_dict(self.config):
            self.config.env.action_space_size = action_space_size
            self.config.env.obs_shape = obs_shape
            self.config.rl.discount **= self.config.env.n_skip
            self.config.model.reward_support.size = reward_size
            self.config.model.value_support.size = value_size
            self.config.save_path += tag

        self.action_space_size = self.config.env.action_space_size
        self.obs_shape = self.config.env.obs_shape
        self.n_stack = self.config.env.n_stack
        self.rep_net_shape = self.config.model.rep_net_shape
        self.hidden_shape = self.config.model.hidden_shape
        self.dyn_shape = self.config.model.dyn_shape
        self.act_embed_shape = self.config.model.act_embed_shape
        self.rew_net_shape = self.config.model.rew_net_shape
        self.val_net_shape = self.config.model.val_net_shape
        self.pi_net_shape = self.config.model.pi_net_shape

        self.proj_hid_shape = self.config.model.proj_hid_shape
        self.pred_hid_shape = self.config.model.pred_hid_shape
        self.proj_shape = self.config.model.proj_shape
        self.pred_shape = self.config.model.pred_shape

        self._update = True
        self.use_bn = self.config.model.use_bn
        self.use_p_norm = self.config.model.use_p_norm
        self.noisy_net = self.config.model.noisy_net

    def build_model(self):

        representation_model = RepresentationNetwork(self.obs_shape, self.n_stack, self.num_blocks,
                                                     self.rep_net_shape, self.hidden_shape,
                                                     use_bn=self.use_bn)
        value_output_size = self.config.model.value_support.size if self.config.model.value_support.type != 'symlog' else 1
        reward_output_size = self.config.model.reward_support.size if self.config.model.reward_support.type != 'symlog' else 1
        dynamics_model = DynamicsNetwork(self.hidden_shape, self.action_space_size, self.num_blocks, self.dyn_shape,
                                         self.act_embed_shape, self.rew_net_shape, reward_output_size,
                                         use_bn=self.use_bn)
        value_policy_model = ValuePolicyNetwork(self.hidden_shape, self.val_net_shape, self.pi_net_shape,
                                                self.action_space_size, value_output_size,
                                                init_zero=self.init_zero, use_bn=self.use_bn, p_norm=self.use_p_norm,
                                                policy_distr=self.config.model.policy_distribution, noisy=self.noisy_net,
                                                value_support=self.config.model.value_support,
                                                v_num=self.v_num)

        if self.config.model.value_prefix:
            reward_prediction_model = RewardNetworkLSTM(self.hidden_shape, self.rew_net_shape, reward_output_size, self.config.model.lstm_hidden_size,
                                                        init_zero=self.init_zero, use_bn=self.use_bn)
        else:
            reward_prediction_model = RewardNetwork(self.hidden_shape, self.rew_net_shape, reward_output_size,
                                                    init_zero=self.init_zero, use_bn=self.use_bn)

        projection_model = nn.Sequential(
            nn.Linear(self.hidden_shape, self.proj_hid_shape),
            nn.LayerNorm(self.proj_hid_shape),
            nn.ReLU(),
            nn.Linear(self.proj_hid_shape, self.proj_hid_shape),
            nn.LayerNorm(self.proj_hid_shape),
            nn.ReLU(),
            nn.Linear(self.proj_hid_shape, self.proj_shape),
            nn.LayerNorm(self.proj_shape)
        )
        projection_head_model = nn.Sequential(
            nn.Linear(self.proj_shape, self.pred_hid_shape),
            nn.LayerNorm(self.pred_hid_shape),
            nn.ReLU(),
            nn.Linear(self.pred_hid_shape, self.pred_shape),
        )

        ez_model = EfficientZero(representation_model, dynamics_model, reward_prediction_model, value_policy_model,
                                 projection_model, projection_head_model, self.config,
                                 state_norm=self.state_norm, value_prefix=self.value_prefix)

        return ez_model
