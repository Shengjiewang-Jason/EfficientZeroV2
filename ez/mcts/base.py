# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import copy
import torch
from torch.cuda.amp import autocast as autocast
from ez.utils.format import DiscreteSupport
import numpy as np


class MCTS:
    def __init__(self, num_actions, **kwargs):
        """

        :param num_actions:
        :param num_top_actions:
        :param kwargs:
        """
        self.num_actions = num_actions

        self.num_simulations = kwargs.get('num_simulations')
        self.num_top_actions = kwargs.get('num_top_actions')
        self.c_visit = kwargs.get('c_visit')
        self.c_scale = kwargs.get('c_scale')
        self.c_base = kwargs.get('c_base')
        self.c_init = kwargs.get('c_init')
        self.dirichlet_alpha = kwargs.get('dirichlet_alpha')
        self.explore_frac = kwargs.get('explore_frac')
        self.discount = kwargs.get('discount')
        self.value_minmax_delta = kwargs.get('value_minmax_delta')
        self.value_support = kwargs.get('value_support')
        self.reward_support = kwargs.get('reward_support')
        self.value_prefix = kwargs.get('value_prefix')
        self.lstm_hidden_size = kwargs.get('lstm_hidden_size')
        self.lstm_horizon_len = kwargs.get('lstm_horizon_len')
        self.mpc_horizon = kwargs.get('mpc_horizon')
        self.env = kwargs.get('env')
        self.vis = kwargs.get('vis')                                    # vis: [log, text, graph]
        self.std_magnification = kwargs.get('std_magnification')

        self.current_num_top_actions = self.num_top_actions             # /2 every phase
        self.current_phase = 0                                          # current phase index
        self.visit_num_for_next_phase = max(
            np.floor(self.num_simulations / (np.log2(self.num_top_actions) * self.current_num_top_actions)), 1) \
                                        * self.current_num_top_actions   # how many visit counts for next phase
        self.used_visit_num = 0
        self.verbose = 0
        assert self.num_top_actions <= self.num_actions

    def search(self, model, batch_size, root_states, root_values, root_policy_logits, **kwargs):
        raise NotImplementedError()

    def sample_mpc_actions(self, policy):
        is_continuous = (self.env in ['DMC', 'Gym'])
        if is_continuous:
            action_dim = policy.shape[-1] // 2
            mean = policy[:, :action_dim]
            return mean
        else:
            return policy.argmax(dim=-1).unsqueeze(1)

    def update_statistics(self, **kwargs):
        if kwargs.get('prediction'):
            # prediction for next states, rewards, values, logits
            model = kwargs.get('model')
            states = kwargs.get('states')
            last_actions = kwargs.get('actions')
            reward_hidden = kwargs.get('reward_hidden')

            next_value_prefixes = 0
            for _ in range(self.mpc_horizon):
                with torch.no_grad():
                    with autocast():
                        states, pred_value_prefixes, next_values, next_logits, reward_hidden = \
                            model.recurrent_inference(states, last_actions, reward_hidden)
                # last_actions = self.sample_mpc_actions(next_logits)
                next_value_prefixes += pred_value_prefixes

            # process outputs
            next_value_prefixes = next_value_prefixes.detach().cpu().numpy()
            next_values = next_values.detach().cpu().numpy()

            self.log('simulate action {}, r = {:.3f}, v = {:.3f}, logits = {}'
                     ''.format(last_actions[0].tolist(), next_value_prefixes[0].item(), next_values[0].item(), next_logits[0].tolist()),
                     verbose=3)
            return states, next_value_prefixes, next_values, next_logits, reward_hidden
        else:
            # env simulation for next states
            env = kwargs.get('env')
            current_states = kwargs.get('states')
            last_actions = kwargs.get('actions')
            states = env.step(last_actions)
            raise NotImplementedError()

    def estimate_value(self, **kwargs):
        # prediction for value in planning
        model = kwargs.get('model')
        current_states = kwargs.get('states')
        actions = kwargs.get('actions')
        reward_hidden = kwargs.get('reward_hidden')

        Value = 0
        discount = 1
        for i in range(actions.shape[0]):
            current_states_hidden = None
            with torch.no_grad():
                with autocast():
                    next_states, next_value_prefixes, next_values, next_logits, reward_hidden = model.recurrent_inference(current_states, actions[i], reward_hidden)

            next_value_prefixes = next_value_prefixes.detach()
            next_values = next_values.detach()
            current_states = next_states
            Value += next_value_prefixes * discount
            discount *= self.discount

        Value += discount * next_values

        return Value

    def log(self, string, verbose, iteration_begin=False, iteration_end=False):
        if verbose <= self.verbose:
            if iteration_begin:
                print('>' * 50)
            print(string)
            print('-' * 20)
            if iteration_end:
                print('<' * 50)

    def reset(self):
        self.current_num_top_actions = self.num_top_actions
        self.current_phase = 0
        self.visit_num_for_next_phase = max(
            np.floor(self.num_simulations / (np.log2(self.num_top_actions) * self.current_num_top_actions)), 1) \
                                        * self.current_num_top_actions  # how many visit counts for next phase
        self.used_visit_num = 0
        self.verbose = 0
