# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import copy
import torch
import torchrl
import numpy as np
import math

from .base import MCTS
# import sys
# sys.path.append('/workspace/EZ-Codebase')
# # import ez.mcts.ctree.cytree as tree
from ez.mcts.ctree import cytree as tree
from ez.mcts.ori_ctree import cytree as ori_tree
from ez.mcts.ctree_v2 import cytree as tree2
from torch.cuda.amp import autocast as autocast
from ez.utils.format import DiscreteSupport, symexp, pad_and_mask
from ez.utils.distribution import SquashedNormal, TruncatedNormal, ContDist
import colorednoise as cn

class CyMCTS(MCTS):
    def __init__(self, num_actions, **kwargs):
        super().__init__(num_actions, **kwargs)
        self.policy_action_num = kwargs.get('policy_action_num')
        self.random_action_num = kwargs.get('random_action_num')
        self.policy_distribution = kwargs.get('policy_distribution')

    def sample_actions(self, policy, add_noise=True, temperature=1.0, input_noises=None, input_dist=None, input_actions=None, sample_nums=None, states=None):
        n_policy = self.policy_action_num
        n_random = self.random_action_num
        if sample_nums:
            n_policy = math.ceil(sample_nums / 2)
            n_random = sample_nums - n_policy
        std_magnification = self.std_magnification
        action_dim = policy.shape[-1] // 2

        if input_dist is not None:
            n_policy //= 2
            n_random //= 2

        Dist = SquashedNormal
        mean, std = policy[:, :action_dim], policy[:, action_dim:]
        distr = Dist(mean, std)
        sampled_actions = distr.sample(torch.Size([n_policy + n_random]))

        policy_actions = sampled_actions[:n_policy]
        random_actions = sampled_actions[-n_random:]

        random_distr = distr
        if add_noise:
            if input_noises is None:
                random_distr = Dist(mean, std_magnification * std)  # more flatten gaussian policy
                random_actions = random_distr.sample(torch.Size([n_random]))
            else:
                noises = torch.from_numpy(input_noises).float().cuda()
                random_actions += noises

        if input_dist is not None:
            refined_mean, refined_std = input_dist[:, :action_dim], input_dist[:, action_dim:]
            refined_distr = Dist(refined_mean, refined_std)
            refined_actions = refined_distr.sample(torch.Size([n_policy + n_random]))

            refined_policy_actions = refined_actions[:n_policy]
            refined_random_actions = refined_actions[-n_random:]

            if add_noise:
                if input_noises is None:
                    refined_random_distr = Dist(refined_mean, std_magnification * refined_std)
                    refined_random_actions = refined_random_distr.sample(torch.Size([n_random]))
                else:
                    noises = torch.from_numpy(input_noises).float().cuda()
                    refined_random_actions += noises

        all_actions = torch.cat((policy_actions, random_actions), dim=0)
        if input_actions is not None:
            all_actions = torch.from_numpy(input_actions).float().cuda()
        if input_dist is not None:
            all_actions = torch.cat((all_actions, refined_policy_actions, refined_random_actions), dim=0)
        all_actions = all_actions.clip(-0.999, 0.999)

        assert (n_policy + n_random) == sample_nums if sample_nums is not None else self.num_actions
        ratio = n_policy / (sample_nums if sample_nums is not None else self.num_actions)
        probs = distr.log_prob(all_actions) - (ratio * distr.log_prob(all_actions) + (1 - ratio) * random_distr.log_prob(all_actions))
        probs = probs.sum(-1).permute(1, 0)
        all_actions = all_actions.permute(1, 0, 2)
        return all_actions, probs


    def inv_softmax(self, dist):
        constant = 100
        return np.log(dist) + constant

    def atanh(self, x):
        return 0.5 * (np.log1p(x) - np.log1p(-x))

    def softmax_temperature(self, dist, temperature=1.0):
        soft_dist = temperature * dist
        dist_max = soft_dist.max(-1, keepdims=True)
        scores = np.exp(soft_dist - dist_max)
        return scores / scores.sum(-1, keepdims=True)

    def q_init(self, states, sampled_actions, model):
        action_num = sampled_actions.shape[1]
        q_inits = []
        for i in range(action_num):
            _, rewards, next_values, _, _ = self.update_statistics(
                prediction=True,  # use model prediction instead of env simulation
                model=model,  # model
                states=states,  # current states
                actions=sampled_actions[:, i],  # last actions
                reward_hidden=None,  # reward hidden
            )
            q_inits.append(rewards + self.discount * next_values)
        return np.asarray(q_inits).swapaxes(0, 1).tolist()


    def search_continuous(self, model, batch_size, root_states, root_values, root_policy_logits,
                          use_gumble_noise=False, temperature=1.0, verbose=0, add_noise=True,
                          input_noises=None, input_dist=None, input_actions=None, prev_mean=None, **kwargs):
        # preparation
        # Node.set_static_attributes(self.discount, self.num_actions)  # set static parameters of MCTS
        # set root nodes for the batch
        # root_sampled_actions, policy_priors = self.sample_actions(root_policy_logits, std_mag=3 if add_noise else 1)
        root_sampled_actions, policy_priors = self.sample_actions(root_policy_logits, add_noise, temperature, input_noises, input_dist=input_dist, input_actions=input_actions)

        sampled_action_num = root_sampled_actions.shape[1]
        uniform_policy = [
            [0.0 for _ in range(sampled_action_num)]
            for _ in range(batch_size)
        ]
        leaf_num = 2
        uniform_policy_non_root = [
            [0.0 for _ in range(leaf_num)] for _ in range(batch_size)
        ]

        # set gumble noise (during training)
        if use_gumble_noise:
            gumble_noises = np.random.gumbel(0, 1, (batch_size, self.num_actions)) #* temperature
        else:
            gumble_noises = np.zeros((batch_size, self.num_actions))

        gumble_noises = gumble_noises.tolist()

        roots = tree.Roots(batch_size, self.num_actions, self.num_simulations, self.discount)
        roots.prepare(root_values.tolist(), uniform_policy, leaf_num)
        # save the min and max value of the tree nodes
        value_min_max_lst = tree.MinMaxStatsList(batch_size)
        value_min_max_lst.set_static_val(self.value_minmax_delta, self.c_visit, self.c_scale)

        reward_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_size).cuda().float(),
                         torch.zeros(1, batch_size, self.lstm_hidden_size).cuda().float())

        # index of states
        state_pool = [root_states]
        hidden_state_index_x = 0
        # 1 x batch x 64
        reward_hidden_c_pool = [reward_hidden[0]]
        reward_hidden_h_pool = [reward_hidden[1]]

        assert batch_size == len(root_states) == len(root_values)
        # expand the roots and update the statistics

        self.verbose = verbose
        if self.verbose:
            np.set_printoptions(precision=3)
            assert batch_size == 1
            self.log('Gumble Noise: {}'.format(gumble_noises), verbose=1)

        # search for N iterations
        mcts_info = {}
        actions_pool = [root_sampled_actions]

        for simulation_idx in range(self.num_simulations):
            current_states = []
            hidden_states_c_reward = []
            hidden_states_h_reward = []
            results = tree.ResultsWrapper(batch_size)

            self.log('Iteration {} \t'.format(simulation_idx), verbose=2, iteration_begin=True)
            if self.verbose > 1:
                self.log('Tree:', verbose=2)
                roots.print_tree()

            # select action for the roots
            hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = \
                tree.batch_traverse(roots, value_min_max_lst, results, self.num_simulations, simulation_idx,
                                    gumble_noises, self.current_num_top_actions)

            search_lens = results.get_search_len()
            selected_actions = []
            ptr = 0
            for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                current_states.append(state_pool[ix][iy])
                if self.value_prefix:
                    hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy])
                    hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy])

                selected_actions.append(actions_pool[ix][iy][last_actions[ptr]])
                ptr += 1

            current_states = torch.stack(current_states)
            if self.value_prefix:
                hidden_states_c_reward = torch.stack(hidden_states_c_reward).unsqueeze(0)
                hidden_states_h_reward = torch.stack(hidden_states_h_reward).unsqueeze(0)
            selected_actions = torch.stack(selected_actions)

            # inference state, reward, value, policy given the current state
            reward_hidden = (hidden_states_c_reward, hidden_states_h_reward)
            mcts_info[simulation_idx] = {
                'states': current_states,
                'actions': last_actions,
                'reward_hidden': reward_hidden,
            }

            next_states, next_value_prefixes, next_values, next_logits, reward_hidden = self.update_statistics(
                prediction=True,                                    # use model prediction instead of env simulation
                model=model,                                        # model
                states=current_states,                              # current states
                actions=selected_actions,                           # last actions
                reward_hidden=reward_hidden,                        # reward hidden
            )
            mcts_info[simulation_idx] = {
                'next_states': next_states,
                'next_value_prefixes': next_value_prefixes,
                'next_values': next_values,
                'next_logits': next_logits,
                'next_reward_hidden': reward_hidden
            }
            leaf_sampled_actions, leaf_policy_priors = self.sample_actions(next_logits, sample_nums=leaf_num, add_noise=False)
            # leaf_sampled_actions, leaf_policy_priors = self.sample_actions(next_logits, sample_nums=leaf_num)

            actions_pool.append(leaf_sampled_actions)
            # save to database
            state_pool.append(next_states)
            # change value prefix to reward
            if self.value_prefix:
                reset_idx = (np.array(search_lens) % self.lstm_horizon_len == 0)
                reward_hidden[0][:, reset_idx, :] = 0
                reward_hidden[1][:, reset_idx, :] = 0
                reward_hidden_c_pool.append(reward_hidden[0])
                reward_hidden_h_pool.append(reward_hidden[1])
            else:
                reset_idx = np.asarray([1. for _ in range(batch_size)])

            to_reset_lst = reset_idx.astype(np.int32).tolist()
            hidden_state_index_x += 1

            # expand the leaf node and backward for statistics update
            tree.batch_back_propagate(hidden_state_index_x, next_value_prefixes.squeeze(-1).tolist(),
                                      next_values.squeeze(-1).tolist(), uniform_policy_non_root, value_min_max_lst,
                                      results, to_reset_lst, leaf_num)

            # sequential halving
            if self.ready_for_next_gumble_phase(simulation_idx):

                tree.batch_sequential_halving(roots, gumble_noises, value_min_max_lst, self.current_phase,
                                              self.current_num_top_actions)
                self.log('change to phase: {}, top m action -> {}'
                         ''.format(self.current_phase, self.current_num_top_actions), verbose=3)

        # obtain the final results and infos
        search_root_values = np.asarray(roots.get_values())
        search_root_policies = np.asarray(roots.get_root_policies(value_min_max_lst))

        search_best_actions = np.asarray(roots.get_best_actions())
        root_sampled_actions = root_sampled_actions.detach().cpu().numpy()
        final_selected_actions = np.asarray(
            [root_sampled_actions[i, best_a] for i, best_a in enumerate(search_best_actions)]
        )

        if self.verbose:
            self.log('Final Tree:', verbose=1)
            roots.print_tree()
            self.log('search root value -> \t\t {} \n'
                     'search root policy -> \t\t {} \n'
                     'search best action -> \t\t {}'
                     ''.format(search_root_values[0], search_root_policies[0], search_best_actions[0]),
                     verbose=1, iteration_end=True)

        return search_root_values, search_root_policies, final_selected_actions, root_sampled_actions, search_best_actions, mcts_info

    def select_action(self, visit_counts, temperature=1, deterministic=False):
        action_probs = visit_counts ** (1 / temperature)
        total_count = action_probs.sum(-1, keepdims=True)
        action_probs = action_probs / total_count
        if deterministic:
            action_pos = action_probs.argmax(-1)
        else:
            action_pos = []
            for i in range(action_probs.shape[0]):
                action_pos.append(np.random.choice(action_probs.shape[1], p=action_probs[i]))
            action_pos = np.asarray(action_pos)
            # action_pos = torch.nn.functional.gumbel_softmax(torch.from_numpy(action_probs), hard=True, dim=1).argmax(-1)

        return action_pos

    def search_ori_mcts(self, model, batch_size, root_states, root_values, root_policy_logits,
                        use_noise=True, temperature=1.0, verbose=0, is_reanalyze=False, **kwargs):
        # preparation
        # set dirichley noise (during training)
        if use_noise:
            noises = np.asarray([np.random.dirichlet([self.dirichlet_alpha] * self.num_actions).astype(np.float32).tolist() for _
                      in range(batch_size)])
        else:
            noises = np.zeros((batch_size, self.num_actions))
        noises = noises.tolist()
        # Node.set_static_attributes(self.discount, self.num_actions)  # set static parameters of MCTS
        # set root nodes for the batch
        roots = ori_tree.Roots(batch_size, self.num_actions, self.num_simulations)
        roots.prepare(self.explore_frac, noises, [0. for _ in range(batch_size)], root_policy_logits.tolist())
        # save the min and max value of the tree nodes
        value_min_max_lst = ori_tree.MinMaxStatsList(batch_size)
        value_min_max_lst.set_delta(self.value_minmax_delta)

        if self.value_prefix:
            reward_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_size).cuda().float(),
                             torch.zeros(1, batch_size, self.lstm_hidden_size).cuda().float())
        else:
            reward_hidden = None

        # index of states
        state_pool = [root_states]
        hidden_state_index_x = 0
        # 1 x batch x 64
        reward_hidden_c_pool = [reward_hidden[0]]
        reward_hidden_h_pool = [reward_hidden[1]]

        assert batch_size == len(root_states) == len(root_values)
        # expand the roots and update the statistics

        self.verbose = verbose
        if self.verbose:
            np.set_printoptions(precision=3)
            assert batch_size == 1
            self.log('Dirichlet Noise: {}'.format(noises), verbose=1)

        # search for N iterations
        mcts_info = {}
        for simulation_idx in range(self.num_simulations):
            current_states = []
            hidden_states_c_reward = []
            hidden_states_h_reward = []
            results = ori_tree.ResultsWrapper(batch_size)

            self.log('Iteration {} \t'.format(simulation_idx), verbose=2, iteration_begin=True)
            if self.verbose > 1:
                self.log('Tree:', verbose=2)
                roots.print_tree()

            # select action for the roots
            hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = ori_tree.batch_traverse(roots, self.c_base, self.c_init, self.discount, value_min_max_lst, results)
            search_lens = results.get_search_len()

            for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                current_states.append(state_pool[ix][iy])
                hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy])
                hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy])

            current_states = torch.stack(current_states)
            hidden_states_c_reward = torch.stack(hidden_states_c_reward).unsqueeze(0)
            hidden_states_h_reward = torch.stack(hidden_states_h_reward).unsqueeze(0)
            last_actions = torch.from_numpy(np.asarray(last_actions)).cuda().long().unsqueeze(1)

            # inference state, reward, value, policy given the current state
            reward_hidden = (hidden_states_c_reward, hidden_states_h_reward)

            next_states, next_value_prefixes, next_values, next_logits, reward_hidden = self.update_statistics(
                prediction=True,                                    # use model prediction instead of env simulation
                model=model,                                        # model
                states=current_states,                              # current states
                actions=last_actions,                               # last actions
                reward_hidden=reward_hidden,                        # reward hidden
            )

            # save to database
            state_pool.append(next_states)
            # change value prefix to reward
            if self.value_prefix:
                reset_idx = (np.array(search_lens) % self.lstm_horizon_len == 0)
                reward_hidden[0][:, reset_idx, :] = 0
                reward_hidden[1][:, reset_idx, :] = 0
                reward_hidden_c_pool.append(reward_hidden[0])
                reward_hidden_h_pool.append(reward_hidden[1])
            else:
                reset_idx = np.asarray([1. for _ in range(batch_size)])
            to_reset_lst = reset_idx.astype(np.int32).tolist()

            hidden_state_index_x += 1

            # expand the leaf node and backward for statistics update
            ori_tree.batch_back_propagate(hidden_state_index_x, self.discount, next_value_prefixes.squeeze(-1).tolist(), next_values.squeeze(-1).tolist(), next_logits.tolist(), value_min_max_lst, results, to_reset_lst)

        # obtain the final results and infos
        search_root_values = np.asarray(roots.get_values())
        search_root_policies = np.asarray(roots.get_distributions())
        if not is_reanalyze:
            search_best_actions = self.select_action(search_root_policies, temperature=temperature, deterministic=not use_noise)
        else:
            search_best_actions = np.zeros(batch_size)

        if self.verbose:
            self.log('Final Tree:', verbose=1)
            roots.print_tree()
            self.log('search root value -> \t\t {} \n'
                     'search root policy -> \t\t {} \n'
                     'search best action -> \t\t {}'
                     ''.format(search_root_values[0], search_root_policies[0], search_best_actions[0]),
                     verbose=1, iteration_end=True)

        search_root_policies = search_root_policies / search_root_policies.sum(-1, keepdims=True)
        return search_root_values, search_root_policies, search_best_actions, mcts_info


    def search(self, model, batch_size, root_states, root_values, root_policy_logits,
               use_gumble_noise=True, temperature=1.0, verbose=0, **kwargs):
        # preparation
        # Node.set_static_attributes(self.discount, self.num_actions)  # set static parameters of MCTS
        # set root nodes for the batch
        roots = tree.Roots(batch_size, self.num_actions, self.num_simulations, self.discount)
        roots.prepare(root_values.tolist(), root_policy_logits.tolist(), self.num_actions)
        # save the min and max value of the tree nodes
        value_min_max_lst = tree.MinMaxStatsList(batch_size)
        value_min_max_lst.set_static_val(self.value_minmax_delta, self.c_visit, self.c_scale)

        if self.value_prefix:
            reward_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_size).cuda().float(),
                             torch.zeros(1, batch_size, self.lstm_hidden_size).cuda().float())
        else:
            reward_hidden = None

        # index of states
        state_pool = [root_states]
        hidden_state_index_x = 0
        # 1 x batch x 64
        reward_hidden_c_pool = [reward_hidden[0]]
        reward_hidden_h_pool = [reward_hidden[1]]

        # set gumble noise (during training)
        if use_gumble_noise:
            gumble_noises = np.random.gumbel(0, 1, (batch_size, self.num_actions)) #* temperature
        else:
            gumble_noises = np.zeros((batch_size, self.num_actions))
        gumble_noises = gumble_noises.tolist()

        assert batch_size == len(root_states) == len(root_values)
        # expand the roots and update the statistics

        self.verbose = verbose
        if self.verbose:
            np.set_printoptions(precision=3)
            assert batch_size == 1
            self.log('Gumble Noise: {}'.format(gumble_noises), verbose=1)


        # search for N iterations
        mcts_info = {}
        for simulation_idx in range(self.num_simulations):
            current_states = []
            hidden_states_c_reward = []
            hidden_states_h_reward = []
            results = tree.ResultsWrapper(batch_size)
            # results1 = tree2.ResultsWrapper(roots1.num)

            self.log('Iteration {} \t'.format(simulation_idx), verbose=2, iteration_begin=True)
            if self.verbose > 1:
                self.log('Tree:', verbose=2)
                roots.print_tree()

            # select action for the roots
            hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = \
                tree.batch_traverse(roots, value_min_max_lst, results, self.num_simulations, simulation_idx,
                                    gumble_noises, self.current_num_top_actions)

            search_lens = results.get_search_len()

            for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                current_states.append(state_pool[ix][iy])
                hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy])
                hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy])

            current_states = torch.stack(current_states)
            hidden_states_c_reward = torch.stack(hidden_states_c_reward).unsqueeze(0)
            hidden_states_h_reward = torch.stack(hidden_states_h_reward).unsqueeze(0)
            last_actions = torch.from_numpy(np.asarray(last_actions)).cuda().long().unsqueeze(1)

            # inference state, reward, value, policy given the current state
            reward_hidden = (hidden_states_c_reward, hidden_states_h_reward)
            mcts_info[simulation_idx] = {
                'states': current_states,
                'actions': last_actions,
                'reward_hidden': reward_hidden,
            }

            next_states, next_value_prefixes, next_values, next_logits, reward_hidden = self.update_statistics(
                prediction=True,                                    # use model prediction instead of env simulation
                model=model,                                        # model
                states=current_states,                              # current states
                actions=last_actions,                               # last actions
                reward_hidden=reward_hidden,                        # reward hidden
            )
            mcts_info[simulation_idx] = {
                'next_states': next_states,
                'next_value_prefixes': next_value_prefixes,
                'next_values': next_values,
                'next_logits': next_logits,
                'next_reward_hidden': reward_hidden
            }

            # save to database
            state_pool.append(next_states)
            # change value prefix to reward
            reset_idx = (np.array(search_lens) % self.lstm_horizon_len == 0)
            if self.value_prefix:
                reward_hidden[0][:, reset_idx, :] = 0
                reward_hidden[1][:, reset_idx, :] = 0
            to_reset_lst = reset_idx.astype(np.int32).tolist()
            if not self.value_prefix:
                to_reset_lst = [1 for _ in range(batch_size)]

            reward_hidden_c_pool.append(reward_hidden[0])
            reward_hidden_h_pool.append(reward_hidden[1])
            hidden_state_index_x += 1

            # expand the leaf node and backward for statistics update
            tree.batch_back_propagate(hidden_state_index_x, next_value_prefixes.squeeze(-1).tolist(), next_values.squeeze(-1).tolist(), next_logits.tolist(), value_min_max_lst, results, to_reset_lst, self.num_actions)


            # sequential halving
            if self.ready_for_next_gumble_phase(simulation_idx):
                tree.batch_sequential_halving(roots, gumble_noises, value_min_max_lst, self.current_phase,
                                              self.current_num_top_actions)
                # if self.current_phase == 0:
                #     search_root_values = np.asarray(roots.get_values())

                self.log('change to phase: {}, top m action -> {}'
                         ''.format(self.current_phase, self.current_num_top_actions), verbose=3)

        # assert self.ready_for_next_gumble_phase(self.num_simulations)
        # final selection
        # tree.batch_sequential_halving(roots, gumble_noises, value_min_max_lst, self.current_phase, self.current_num_top_actions)

        # obtain the final results and infos
        search_root_values = np.asarray(roots.get_values())
        search_root_policies = np.asarray(roots.get_root_policies(value_min_max_lst))
        search_best_actions = np.asarray(roots.get_best_actions())

        if self.verbose:
            self.log('Final Tree:', verbose=1)
            roots.print_tree()
            self.log('search root value -> \t\t {} \n'
                     'search root policy -> \t\t {} \n'
                     'search best action -> \t\t {}'
                     ''.format(search_root_values[0], search_root_policies[0], search_best_actions[0]),
                     verbose=1, iteration_end=True)

 
        return search_root_values, search_root_policies, search_best_actions, mcts_info

    def ready_for_next_gumble_phase(self, simulation_idx):
        ready = (simulation_idx + 1) >= self.visit_num_for_next_phase
        if ready:
            # change the current top action num from m -> m / 2
            self.current_phase += 1
            self.current_num_top_actions = self.current_num_top_actions // 2
            assert self.current_num_top_actions == self.num_top_actions // (2 ** self.current_phase)

            # update the total visit num for the next phase
            n = self.num_simulations
            m = self.num_top_actions
            current_m = self.current_num_top_actions
            # visit n / log2(m) * current_m at current phase
            if current_m > 2:
                extra_visit = max(np.floor(n / (np.log2(m) * current_m)), 1) * current_m
            else:
                extra_visit = n - self.used_visit_num
            self.used_visit_num += extra_visit
            self.visit_num_for_next_phase += extra_visit
            self.visit_num_for_next_phase = min(self.visit_num_for_next_phase, self.num_simulations)

            self.log('Be ready for the next gumble phase at iteration {}: \n'
                     'current top action num is {}, visit {} times for next phase'
                     ''.format(simulation_idx, current_m, self.visit_num_for_next_phase), verbose=3)
        return ready


"""
legacy code of Gumbel search
"""
class Gumbel_MCTS(object):
    def __init__(self, config):
        self.config = config
        self.value_prefix = self.config.model.value_prefix
        self.num_simulations = self.config.mcts.num_simulations
        self.num_top_actions = self.config.mcts.num_top_actions
        self.c_visit = self.config.mcts.c_visit
        self.c_scale = self.config.mcts.c_scale
        self.discount = self.config.rl.discount
        self.value_minmax_delta = self.config.mcts.value_minmax_delta
        self.lstm_hidden_size = self.config.model.lstm_hidden_size
        self.action_space_size = self.config.env.action_space_size
        try:
            self.policy_distribution = self.config.model.policy_distribution
        except:
            pass

    def update_statistics(self, **kwargs):
        if kwargs.get('prediction'):
            # prediction for next states, rewards, values, logits
            model = kwargs.get('model')
            current_states = kwargs.get('states')
            last_actions = kwargs.get('actions')
            reward_hidden = kwargs.get('reward_hidden')

            with torch.no_grad():
                with autocast():
                    next_states, next_value_prefixes, next_values, next_logits, reward_hidden = \
                        model.recurrent_inference(current_states, last_actions, reward_hidden)

            # process outputs
            next_values = next_values.detach().cpu().numpy().flatten()
            next_value_prefixes = next_value_prefixes.detach().cpu().numpy().flatten()
            # if masks is not None:
            #     next_states = next_states[:, -1]
            return next_states, next_value_prefixes, next_values, next_logits, reward_hidden
        else:
            # env simulation for next states
            env = kwargs.get('env')
            current_states = kwargs.get('states')
            last_actions = kwargs.get('actions')
            states = env.step(last_actions)
            raise NotImplementedError()

    def sample_actions(self, policy, add_noise=True, temperature=1.0, input_noises=None, input_dist=None, input_actions=None):
        batch_size = policy.shape[0]
        n_policy = self.config.model.policy_action_num
        n_random = self.config.model.random_action_num
        std_magnification = self.config.mcts.std_magnification
        action_dim = policy.shape[-1] // 2

        if input_dist is not None:
            n_policy //= 2
            n_random //= 2

        Dist = SquashedNormal
        mean, std = policy[:, :action_dim], policy[:, action_dim:]
        distr = Dist(mean, std)
        sampled_actions = distr.sample(torch.Size([n_policy + n_random]))
        sampled_actions = sampled_actions.permute(1, 0, 2)

        policy_actions = sampled_actions[:, :n_policy]
        random_actions = sampled_actions[:, -n_random:]

        if add_noise:
            if input_noises is None:
                # random_distr = Dist(mean, self.std_magnification * std * temperature)       # more flatten gaussian policy
                random_distr = Dist(mean, std_magnification * std)  # more flatten gaussian policy
                random_actions = random_distr.sample(torch.Size([n_random]))
                random_actions = random_actions.permute(1, 0, 2)

                # random_actions = torch.rand(batch_size, n_random, action_dim).float().cuda()
                # random_actions = 2 * random_actions - 1

                # Gaussian noise
                # random_actions += torch.randn_like(random_actions)
            else:
                noises = torch.from_numpy(input_noises).float().cuda()
                random_actions += noises

        if input_dist is not None:
            refined_mean, refined_std = input_dist[:, :action_dim], input_dist[:, action_dim:]
            refined_distr = Dist(refined_mean, refined_std)
            refined_actions = refined_distr.sample(torch.Size([n_policy + n_random]))
            refined_actions = refined_actions.permute(1, 0, 2)

            refined_policy_actions = refined_actions[:, :n_policy]
            refined_random_actions = refined_actions[:, -n_random:]

            if add_noise:
                if input_noises is None:
                    refined_random_distr = Dist(refined_mean, std_magnification * refined_std)
                    refined_random_actions = refined_random_distr.sample(torch.Size([n_random]))
                    refined_random_actions = refined_random_actions.permute(1, 0, 2)
                else:
                    noises = torch.from_numpy(input_noises).float().cuda()
                    refined_random_actions += noises

        all_actions = torch.cat((policy_actions, random_actions), dim=1)
        if input_actions is not None:
            all_actions = torch.from_numpy(input_actions).float().cuda()
        if input_dist is not None:
            all_actions = torch.cat((all_actions, refined_policy_actions, refined_random_actions), dim=1)
        # all_actions[:, 0, :] = mean     # add mean as one of candidate
        all_actions = all_actions.clip(-0.999, 0.999)

        return all_actions

    @torch.no_grad()
    def run_multi_discrete(
        self, model, batch_size,
        hidden_state_roots, root_values,
        root_policy_logits, temperature=1.0,
        use_gumbel_noise=True
    ):

        model.eval()

        reward_sum_pool = [0. for _ in range(batch_size)]

        roots = tree2.Roots(
            batch_size, self.action_space_size,
            self.num_simulations
        )
        root_policy_logits = root_policy_logits.detach().cpu().numpy()

        roots.prepare(
            reward_sum_pool, root_policy_logits.tolist(),
            self.num_top_actions, self.num_simulations,
            root_values.tolist()
        )

        reward_hidden_roots = (
            torch.from_numpy(np.zeros((1, batch_size, self.lstm_hidden_size))).float().cuda(),
            torch.from_numpy(np.zeros((1, batch_size, self.lstm_hidden_size))).float().cuda()
        )

        gumbels = np.random.gumbel(
            0, 1, (batch_size, self.action_space_size)
        )# * temperature
        if not use_gumbel_noise:
            gumbels = np.zeros_like(gumbels)
        gumbels = gumbels.tolist()

        num = roots.num
        c_visit, c_scale, discount = self.c_visit, self.c_scale, self.discount
        hidden_state_pool = [hidden_state_roots]
        # 1 x batch x 64
        reward_hidden_c_pool = [reward_hidden_roots[0]]
        reward_hidden_h_pool = [reward_hidden_roots[1]]
        hidden_state_index_x = 0
        min_max_stats_lst = tree2.MinMaxStatsList(num)
        min_max_stats_lst.set_delta(self.value_minmax_delta)
        horizons = self.config.model.lstm_horizon_len

        for index_simulation in range(self.num_simulations):
            hidden_states = []
            hidden_states_c_reward = []
            hidden_states_h_reward = []

            results = tree2.ResultsWrapper(num)
            hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, _, _, _ = \
                tree2.multi_traverse(
                    roots, c_visit, c_scale, discount,
                    min_max_stats_lst, results,
                    index_simulation, gumbels,
                    # int(self.config.model.dynamic_type == 'Transformer')
                    int(False)
                )
            search_lens = results.get_search_len()

            for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                hidden_states.append(hidden_state_pool[ix][iy].unsqueeze(0))
                if self.value_prefix:
                    hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy].unsqueeze(0))
                    hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy].unsqueeze(0))

            hidden_states = torch.cat(hidden_states, dim=0)
            if self.value_prefix:
                hidden_states_c_reward = torch.cat(hidden_states_c_reward).unsqueeze(0)
                hidden_states_h_reward = torch.cat(hidden_states_h_reward).unsqueeze(0)

            last_actions = torch.from_numpy(
                np.asarray(last_actions)
            ).to('cuda').unsqueeze(1).long()

            hidden_state_nodes, reward_sum_pool, value_pool, policy_logits_pool, reward_hidden_nodes = \
                self.update_statistics(
                    prediction=True,  # use model prediction instead of env simulation
                    model=model,  # model
                    states=hidden_states,  # current states
                    actions=last_actions,  # last actions
                    reward_hidden=(hidden_states_c_reward, hidden_states_h_reward),  # reward hidden
                )

            reward_sum_pool = reward_sum_pool.tolist()
            value_pool = value_pool.tolist()
            policy_logits_pool = policy_logits_pool.detach().cpu().numpy().tolist()

            hidden_state_pool.append(hidden_state_nodes)
            # reset 0
            if self.value_prefix:
                if horizons > 0:
                    reset_idx = (np.array(search_lens) % horizons == 0)
                    assert len(reset_idx) == num
                    reward_hidden_nodes[0][:, reset_idx, :] = 0
                    reward_hidden_nodes[1][:, reset_idx, :] = 0
                    is_reset_lst = reset_idx.astype(np.int32).tolist()
                else:
                    is_reset_lst = [0 for _ in range(num)]
            else:
                is_reset_lst = [1 for _ in range(num)]

            if self.value_prefix:
                reward_hidden_c_pool.append(reward_hidden_nodes[0])
                reward_hidden_h_pool.append(reward_hidden_nodes[1])
            hidden_state_index_x += 1

            tree2.multi_back_propagate(
                hidden_state_index_x, discount,
                reward_sum_pool, value_pool, policy_logits_pool,
                min_max_stats_lst, results, is_reset_lst,
                index_simulation, gumbels, c_visit, c_scale, self.num_simulations
            )

        root_values = np.asarray(roots.get_values())
        pi_primes = np.asarray(roots.get_pi_primes(
            min_max_stats_lst, c_visit, c_scale, discount
        ))
        best_actions = np.asarray(roots.get_actions(
            min_max_stats_lst, c_visit, c_scale, gumbels, discount
        ))
        root_sampled_actions = np.expand_dims(
            np.arange(self.action_space_size), axis=0
        ).repeat(batch_size, axis=0)

        advantages = np.asarray(roots.get_advantages(discount))

        worst_actions = np.asarray(pi_primes).argmin(-1)

        # import ipdb
        # ipdb.set_trace()
        # if best_actions[0] != np.asarray(pi_primes)[0].argmax():
        #     import ipdb
        #     ipdb.set_trace()
        #     print(f'best_actions={best_actions[0]}, largest_i={np.asarray(pi_primes)[0].argmax()}, pi={pi_primes[0]}')

        return root_values, pi_primes, best_actions, \
               min_max_stats_lst.get_min_max(), root_sampled_actions


    def run_multi_continuous(
            self, model, batch_size,
            hidden_state_roots, root_values,
            root_policy_logits, is_reanalyze=False, cnt=-1, temperature=1.0, add_noise=True, use_gumbel_noise=False,
            input_noises=None, input_actions=None
    ):
        with torch.no_grad():
            model.eval()

            reward_sum_pool = [0. for _ in range(batch_size)]
            action_pool = []
            reward_hidden_roots = (
                torch.from_numpy(np.zeros((1, batch_size, self.lstm_hidden_size))).float().cuda(),
                torch.from_numpy(np.zeros((1, batch_size, self.lstm_hidden_size))).float().cuda()
            )
            root_sampled_actions = self.sample_actions(root_policy_logits, add_noise, temperature, input_noises, input_actions=input_actions)
            sampled_action_num = root_sampled_actions.shape[1]

            roots = tree2.Roots(
                batch_size, sampled_action_num, self.num_simulations
            )
            action_pool.append(root_sampled_actions)

            uniform_policy = [
                # [1 / sampled_action_num for _ in range(sampled_action_num)]
                [0.0 for _ in range(sampled_action_num)]
                for _ in range(batch_size)
            ]
            q_inits = uniform_policy
            assert self.num_top_actions == self.config.model.policy_action_num + self.config.model.random_action_num
            # assert np.array(uniform_policy).shape == np.array(eval_policy).shape

            # roots.prepare_q_init(
            #     reward_sum_pool,
            #     uniform_policy,
            #     # eval_policy,
            #     self.num_top_actions,
            #     self.num_simulations,
            #     root_values.tolist(),
            #     # q_inits.tolist()
            #     q_inits
            # )
            roots.prepare(
                reward_sum_pool,
                uniform_policy,
                self.num_top_actions,
                self.num_simulations,
                root_values.tolist()
            )

            gumbels = np.random.gumbel(
                0, 1, (batch_size, sampled_action_num)
            ) * temperature
            if not use_gumbel_noise:
                gumbels = np.zeros_like(gumbels)
            gumbels = gumbels.tolist()

            num = roots.num
            c_visit, c_scale, discount = self.c_visit, self.c_scale, self.discount
            hidden_state_pool = [hidden_state_roots]
            # 1 x batch x 64
            reward_hidden_c_pool = [reward_hidden_roots[0]]
            reward_hidden_h_pool = [reward_hidden_roots[1]]
            hidden_state_index_x = 0
            min_max_stats_lst = tree2.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.value_minmax_delta)
            horizons = self.config.model.lstm_horizon_len

            actions_pool = [root_sampled_actions]
            for index_simulation in range(self.num_simulations):
                hidden_states = []
                states_hidden_c = []
                states_hidden_h = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                results = tree2.ResultsWrapper(num)
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, _, _, _ = \
                    tree2.multi_traverse(roots, c_visit, c_scale, discount, min_max_stats_lst,
                                         results, index_simulation, gumbels, int(self.config.model.dynamic_type == 'Transformer'))
                search_lens = results.get_search_len()

                ptr = 0
                selected_actions = []
                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    hidden_states.append(hidden_state_pool[ix][iy].unsqueeze(0))
                    if self.value_prefix:
                        hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy].unsqueeze(0))
                        hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy].unsqueeze(0))
                    selected_actions.append(
                        actions_pool[ix][iy][last_actions[ptr]].unsqueeze(0)
                    )
                    ptr += 1

                hidden_states = torch.cat(hidden_states, dim=0).float()
                if self.value_prefix:
                    hidden_states_c_reward = torch.cat(hidden_states_c_reward, dim=0).unsqueeze(0)
                    hidden_states_h_reward = torch.cat(hidden_states_h_reward, dim=0).unsqueeze(0)

                selected_actions = torch.cat(selected_actions, dim=0).float()
                hidden_state_nodes, reward_sum_pool, value_pool, policy_logits_pool, reward_hidden_nodes = self.update_statistics(
                    prediction=True,  # use model prediction instead of env simulation
                    model=model,  # model
                    states=hidden_states,  # current states
                    actions=selected_actions,  # last actions
                    reward_hidden=(hidden_states_c_reward, hidden_states_h_reward),  # reward hidden
                )

                leaf_sampled_actions = self.sample_actions(policy_logits_pool, False, input_actions=input_actions)
                actions_pool.append(leaf_sampled_actions)
                reward_sum_pool = reward_sum_pool.tolist()
                value_pool = value_pool.tolist()

                hidden_state_pool.append(hidden_state_nodes)
                # reset 0
                if self.value_prefix:
                    if horizons > 0:
                        reset_idx = (np.array(search_lens) % horizons == 0)
                        assert len(reset_idx) == num
                        reward_hidden_nodes[0][:, reset_idx, :] = 0
                        reward_hidden_nodes[1][:, reset_idx, :] = 0
                        is_reset_lst = reset_idx.astype(np.int32).tolist()
                    else:
                        is_reset_lst = [0 for _ in range(num)]
                else:
                    is_reset_lst = [1 for _ in range(num)]      # TODO: this is a huge bug, previous 0.

                if self.value_prefix:
                    reward_hidden_c_pool.append(reward_hidden_nodes[0])
                    reward_hidden_h_pool.append(reward_hidden_nodes[1])
                hidden_state_index_x += 1

                tree2.multi_back_propagate(
                    hidden_state_index_x, discount,
                    reward_sum_pool, value_pool,
                    uniform_policy,
                    min_max_stats_lst, results, is_reset_lst,
                    index_simulation, gumbels, c_visit, c_scale, self.num_simulations
                )

        best_actions = roots.get_actions(min_max_stats_lst, c_visit, c_scale, gumbels, discount)
        root_sampled_actions = root_sampled_actions.detach().cpu().numpy()
        final_selected_actions = np.asarray(
            [root_sampled_actions[i, best_a] for i, best_a in enumerate(best_actions)]
        )
        advantages = np.asarray(roots.get_advantages(discount))

        # pi_prime = roots.get_pi_primes(min_max_stats_lst, c_visit, c_scale, discount)
        # if best_actions[0] != np.asarray(pi_prime)[0].argmax():
        #     import ipdb
        #     ipdb.set_trace()
        #     print(f'best_actions={best_actions[0]}, largest_i={np.asarray(pi_prime)[0].argmax()}, pi={pi_prime[0]}')

        return np.asarray(roots.get_values()), \
               np.asarray(roots.get_pi_primes(min_max_stats_lst, c_visit, c_scale, discount)), \
               np.asarray(final_selected_actions), min_max_stats_lst.get_min_max(), \
               np.asarray(root_sampled_actions), np.asarray(best_actions)