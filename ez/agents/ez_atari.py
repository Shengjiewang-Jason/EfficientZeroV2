# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import random
import time
import copy
import math
from ez.agents.base import Agent
from omegaconf import open_dict

from ez.envs import make_atari
from ez.utils.format import DiscreteSupport
from ez.agents.models import EfficientZero
from ez.agents.models.base_model import *


class EZAtariAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

        self.update_config()

        self.num_blocks = config.model.num_blocks
        self.num_channels = config.model.num_channels
        self.reduced_channels = config.model.reduced_channels
        self.fc_layers = config.model.fc_layers
        self.down_sample = config.model.down_sample
        self.state_norm = config.model.state_norm
        self.value_prefix = config.model.value_prefix
        self.init_zero = config.model.init_zero
        self.action_embedding = config.model.action_embedding
        self.action_embedding_dim = config.model.action_embedding_dim
        self.value_policy_detach = config.train.value_policy_detach

    def update_config(self):
        assert not self._update

        env = make_atari(self.config.env.game, seed=0, save_path=None, **self.config.env)
        action_space_size = env.action_space.n

        obs_channel = 1 if self.config.env.gray_scale else 3

        reward_support = DiscreteSupport(self.config)
        reward_size = reward_support.size

        value_support = DiscreteSupport(self.config)
        value_size = value_support.size

        localtime = time.strftime('%Y-%m-%d %H:%M:%S')
        tag = '{}-seed={}-{}/'.format(self.config.tag, self.config.env.base_seed, localtime)

        with open_dict(self.config):
            self.config.env.action_space_size = action_space_size
            self.config.mcts.num_top_actions = min(action_space_size, self.config.mcts.num_top_actions)
            self.config.env.obs_shape[0] = obs_channel
            self.config.rl.discount **= self.config.env.n_skip
            self.config.model.reward_support.size = reward_size
            self.config.model.value_support.size = value_size

            if action_space_size < 4:
                self.config.mcts.num_top_actions = 2
                self.config.mcts.num_simulations = 4
            elif action_space_size < 16:
                self.config.mcts.num_top_actions = 4
            else:
                self.config.mcts.num_top_actions = 8

            if not self.config.mcts.use_gumbel:
                self.config.mcts.num_simulations = 50
            print(f'env={self.config.env.env}, game={self.config.env.game}, |A|={action_space_size}, '
                  f'top_m={self.config.mcts.num_top_actions}, N={self.config.mcts.num_simulations}')
            self.config.save_path += tag

        self.obs_shape = copy.deepcopy(self.config.env.obs_shape)
        self.input_shape = copy.deepcopy(self.config.env.obs_shape)
        self.input_shape[0] *= self.config.env.n_stack
        self.action_space_size = self.config.env.action_space_size

        self._update = True

    def build_model(self):
        if self.down_sample:
            state_shape = (self.num_channels, math.ceil(self.obs_shape[1] / 16), math.ceil(self.obs_shape[2] / 16))
        else:
            state_shape = (self.num_channels, self.obs_shape[1], self.obs_shape[2])

        state_dim = state_shape[0] * state_shape[1] * state_shape[2]
        flatten_size = self.reduced_channels * state_shape[1] * state_shape[2]

        representation_model = RepresentationNetwork(self.input_shape, self.num_blocks, self.num_channels, self.down_sample)

        dynamics_model = DynamicsNetwork(self.num_blocks, self.num_channels, self.action_space_size,
                                         action_embedding=self.action_embedding, action_embedding_dim=self.action_embedding_dim)

        value_policy_model = ValuePolicyNetwork(self.num_blocks, self.num_channels, self.reduced_channels, flatten_size,
                                                     self.fc_layers, self.config.model.value_support.size,
                                                     self.action_space_size, self.init_zero,
                                                     value_policy_detach=self.value_policy_detach,
                                                     v_num=self.config.train.v_num)

        reward_output_size = self.config.model.reward_support.size
        if self.value_prefix:
            reward_prediction_model = SupportLSTMNetwork(0, self.num_channels, self.reduced_channels,
                                           flatten_size, self.fc_layers, reward_output_size,
                                           self.config.model.lstm_hidden_size, self.init_zero)
        else:
            reward_prediction_model = SupportNetwork(self.num_blocks, self.num_channels, self.reduced_channels,
                                           flatten_size, self.fc_layers, reward_output_size,
                                           self.init_zero)

        projection_layers = self.config.model.projection_layers
        head_layers = self.config.model.prjection_head_layers
        assert projection_layers[1] == head_layers[1]

        projection_model = ProjectionNetwork(state_dim, projection_layers[0], projection_layers[1])
        projection_head_model = ProjectionHeadNetwork(projection_layers[1], head_layers[0], head_layers[1])

        ez_model = EfficientZero(representation_model, dynamics_model, reward_prediction_model, value_policy_model,
                                 projection_model, projection_head_model, self.config,
                                 state_norm=self.state_norm, value_prefix=self.value_prefix)

        return ez_model
