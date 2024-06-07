# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from .base import MCTS
from ez.utils.format import softmax
from ez.utils.distribution import SquashedNormal


class MinMaxStats:
    def __init__(self, minmax_delta, min_value_bound=None, max_value_bound=None):
        """
        Minimum and Maximum statistics
        :param minmax_delta: float, for soft update
        :param min_value_bound:
        :param max_value_bound:
        """
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')
        self.minmax_delta = minmax_delta

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            if value >= self.maximum:
                value = self.maximum
            elif value <= self.minimum:
                value = self.minimum
            # We normalize only when we have set the maximum and minimum values.
            value = (value - self.minimum) / max(self.maximum - self.minimum, self.minmax_delta)  # [-1, 1] range

        value = max(min(value, 1), 0)
        return value

    def clear(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')


class Node:
    discount = 0
    num_actions = 0

    @staticmethod
    def set_static_attributes(discount, num_actions):
        Node.discount = discount
        Node.num_actions = num_actions

    def __init__(self, prior, action=None, parent=None):
        self.prior = prior
        self.action = action
        self.parent = parent

        self.depth = parent.depth + 1 if parent else 0
        self.visit_count = 0
        self.value_prefix = 0.

        self.state = None
        self.reward_hidden = None
        self.estimated_value_lst = []
        self.children = []
        self.selected_children_idx = []
        self.reset_value_prefix = True

        self.epsilon = 1e-6

        assert Node.num_actions > 1
        assert 0 < Node.discount <= 1.

    def expand(self, state, value_prefix, policy_logits, reward_hidden=None, reset_value_prefix=True):
        self.state = state
        self.reward_hidden = reward_hidden
        self.value_prefix = value_prefix
        self.reset_value_prefix = reset_value_prefix

        for action in range(Node.num_actions):
            prior = policy_logits[action]
            child = Node(prior, action, self)

            self.children.append(child)

    def get_policy(self):
        logits = np.asarray([child.prior for child in self.children])
        return softmax(logits)

    def get_improved_policy(self, transformed_completed_Qs):
        logits = np.asarray([child.prior for child in self.children])
        return softmax(logits + transformed_completed_Qs)

    def get_v_mix(self):
        """
        v_mix implementation, refer to https://openreview.net/pdf?id=bERaNdoegnO (Appendix D)
        """
        pi_lst = self.get_policy()
        pi_sum = 0
        pi_qsa_sum = 0

        for action, child in enumerate(self.children):
            if child.is_expanded():
                pi_sum += pi_lst[action]
                pi_qsa_sum += pi_lst[action] * self.get_qsa(action)

        # if no child has been visited
        if pi_sum < self.epsilon:
            v_mix = self.get_value()
        else:
            visit_sum = self.get_children_visit_sum()
            v_mix = (1. / (1. + visit_sum)) * (self.get_value() + visit_sum * pi_qsa_sum / pi_sum)

        return v_mix

    def get_completed_Q(self, normalize_func):
        completed_Qs = []
        v_mix = self.get_v_mix()
        for action, child in enumerate(self.children):
            if child.is_expanded():
                completed_Q = self.get_qsa(action)
            else:
                completed_Q = v_mix
            # normalization
            completed_Qs.append(normalize_func(completed_Q))
        return np.asarray(completed_Qs)

    def get_children_priors(self):
        return np.asarray([child.prior for child in self.children])

    def get_children_visits(self):
        return np.asarray([child.visit_count for child in self.children])

    def get_children_visit_sum(self):
        visit_lst = self.get_children_visits()
        visit_sum = np.sum(visit_lst)
        assert visit_sum == self.visit_count - 1
        return visit_sum

    def get_value(self):
        if self.is_expanded():
            return np.mean(self.estimated_value_lst)
        else:
            return self.parent.get_v_mix()

    def get_qsa(self, action):
        child = self.children[action]
        assert child.is_expanded()
        qsa = child.get_reward() + Node.discount * child.get_value()
        return qsa

    def get_reward(self):
        if self.reset_value_prefix:
            return self.value_prefix
        else:
            assert self.parent is not None
            return self.value_prefix - self.parent.value_prefix

    def get_root(self):
        node = self
        while not node.is_root():
            node = node.parent
        return node

    def get_expanded_children(self):
        assert self.is_expanded()

        children = []
        for _, child in enumerate(self.children):
            if child.is_expanded():
                children.append(child)
        return children

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        assert self.is_expanded()
        return len(self.get_expanded_children()) == 0

    def is_expanded(self):
        assert (len(self.children) > 0) == (self.visit_count > 0)
        return len(self.children) > 0

    def print(self, info):
        if not self.is_expanded():
            return

        for i in range(self.depth):
            print(info[i], end='')

        is_leaf = self.is_leaf()
        if is_leaf:
            print('└──', end='')
        else:
            print('├──', end='')

        print(self.__str__())

        for child in self.get_expanded_children():
            c = '   ' if is_leaf else '|    '
            info.append(c)
            child.print(info)

    def __str__(self):
        if self.is_root():
            action = self.selected_children_idx
        else:
            action = self.action

        s = '[a={} reset={} (n={}, vp={:.3f} r={:.3f}, v={:.3f})]' \
            ''.format(action, self.reset_value_prefix, self.visit_count, self.value_prefix, self.get_reward(), self.get_value())
        return s


class PyMCTS(MCTS):
    def __init__(self, num_actions, **kwargs):
        super().__init__(num_actions, **kwargs)

    def sample_actions(self, policy, add_noise=True, temperature=1.0, input_noises=None, input_dist=None, input_actions=None):
        batch_size = policy.shape[0]
        n_policy = self.policy_action_num
        n_random = self.random_action_num
        std_magnification = self.std_magnification
        action_dim = policy.shape[-1] // 2

        if input_dist is not None:
            n_policy //= 2
            n_random //= 2

        Dist = SquashedNormal
        mean, std = policy[:, :action_dim], policy[:, action_dim:]
        distr = Dist(mean, std)
        # distr = ContDist(torch.distributions.independent.Independent(torch.distributions.normal.Normal(mean, std), 1))
        sampled_actions = distr.sample(torch.Size([n_policy + n_random]))
        sampled_actions = sampled_actions.permute(1, 0, 2)

        policy_actions = sampled_actions[:, :n_policy]
        random_actions = sampled_actions[:, -n_random:]

        if add_noise:
            if input_noises is None:
                # random_distr = Dist(mean, self.std_magnification * std * temperature)       # more flatten gaussian policy
                random_distr = Dist(mean, std_magnification * std)  # more flatten gaussian policy
                # random_distr = ContDist(
                #     torch.distributions.independent.Independent(torch.distributions.normal.Normal(mean, std_magnification * std), 1))
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

        # probs = distr.log_prob(all_actions.permute(1, 0, 2)).exp().mean(-1).permute(1, 0)
        probs = None
        return all_actions, probs

    def search_continuous(self, model, batch_size, root_states, root_values, root_policy_logits,
                          use_gumble_noise=False, temperature=1.0, verbose=0, **kwargs):

        root_sampled_actions, policy_priors = self.sample_actions(root_policy_logits, True, temperature,
                                                                  None, input_dist=None,
                                                                  input_actions=None)
        sampled_action_num = root_sampled_actions.shape[1]
        # preparation
        Node.set_static_attributes(self.discount, self.num_actions)  # set static parameters of MCTS
        roots = [Node(prior=1) for _ in range(batch_size)]          # set root nodes for the batch
        # expand the roots and update the statistics
        for root, state, value, logit in zip(roots, root_states, root_values, root_policy_logits):
            root_reward_hidden = (torch.zeros(1, self.lstm_hidden_size).cuda().float(),
                                  torch.zeros(1, self.lstm_hidden_size).cuda().float())
            if not self.value_prefix:
                root_reward_hidden = None

            root.expand(state, 0, logit, reward_hidden=root_reward_hidden)
            root.estimated_value_lst.append(value)
            root.visit_count += 1
        # save the min and max value of the tree nodes
        value_min_max_lst = [MinMaxStats(self.value_minmax_delta) for _ in range(batch_size)]

        # set gumble noise (during training)
        if use_gumble_noise:
            gumble_noises = np.random.gumbel(0, 1, (batch_size, self.num_actions)) * temperature
        else:
            gumble_noises = np.zeros((batch_size, self.num_actions))

        assert batch_size == len(root_states) == len(root_values)
        self.verbose = verbose
        if self.verbose:
            np.set_printoptions(precision=3)
            assert batch_size == 1

            self.log('Gumble Noise: {}'.format(gumble_noises), verbose=1)

        action_pool = [root_sampled_actions]
        # search for N iterations
        mcts_info = {}
        for simulation_idx in range(self.num_simulations):
            leaf_nodes = []                                         # leaf node of the tree of the current simulation
            last_actions = []                                       # the chosen action of the leaf node
            current_states = []                                     # the hidden state of the leaf node
            reward_hidden = ([], [])                        # the reward hidden of lstm
            search_paths = []                                       # the nodes along the current search iteration

            self.log('Iteration {} \t'.format(simulation_idx), verbose=2, iteration_begin=True)
            if self.verbose > 1:
                self.log('Tree:', verbose=2)
                roots[0].print([])

            # select action for the roots
            trajectories = []
            for idx in range(batch_size):
                node = roots[idx]                                   # search begins from the root node
                search_path = [node]                                # save the search path from root to leaf
                value_min_max = value_min_max_lst[idx]              # record the min, max value of the tree states

                # search from the root until a leaf unexpanded node
                action = -1
                select_action_lst = []
                while node.is_expanded():
                    action = self.select_action(node, value_min_max, gumble_noises, simulation_idx)
                    node = node.children[action]
                    search_path.append(node)
                    select_action_lst.append(action)

                # assert action >= 0

                self.log('selection path -> {}'.format(select_action_lst), verbose=4)
                # update some statistics
                parent = search_path[-2]                            # get the parent of the leaf node
                current_states.append(parent.state)
                reward_hidden[0].append(parent.reward_hidden[0])
                reward_hidden[1].append(parent.reward_hidden[1])

                last_actions.append(action_pool[-1][action])
                leaf_nodes.append(node)
                search_paths.append(search_path)
                trajectories.append(select_action_lst)

            # inference state, reward, value, policy given the current state
            current_states = torch.stack(current_states, dim=0)
            reward_hidden = (torch.stack(reward_hidden[0], dim=1),
                             torch.stack(reward_hidden[1], dim=1))
            last_actions = torch.from_numpy(np.asarray(last_actions)).cuda().long().unsqueeze(1)
            next_states, next_value_prefixes, next_values, next_logits, reward_hidden = self.update_statistics(
                prediction=True,                                    # use model prediction instead of env simulation
                model=model,                                        # model
                states=current_states,                              # current states
                actions=last_actions,                               # last actions
                reward_hidden=reward_hidden,                        # reward hidden
            )

            leaf_sampled_actions, leaf_policy_priors = \
                self.sample_actions(next_logits, add_noise=False,
                                    # input_actions=root_sampled_actions.detach().cpu().numpy()     # FOR TEST SEARCH ALIGHMENT ONLY !!
                                    )
            action_pool.append(leaf_sampled_actions)

            # change value prefix to reward
            search_lens = [len(search_path) for search_path in search_paths]
            reset_idx = (np.array(search_lens) % self.lstm_horizon_len == 0)
            if self.value_prefix:
                reward_hidden[0][:, reset_idx, :] = 0
                reward_hidden[1][:, reset_idx, :] = 0
            to_reset_lst = reset_idx.astype(np.int32).tolist()

            # expand the leaf node and backward for statistics update
            for idx in range(batch_size):
                # expand the leaf node
                leaf_nodes[idx].expand(next_states[idx], next_value_prefixes[idx], next_logits[idx],
                                       (reward_hidden[0][0][idx].unsqueeze(0), reward_hidden[1][0][idx].unsqueeze(0)),
                                       to_reset_lst[idx])
                # backward from the leaf node to the root
                self.back_propagate(search_paths[idx], next_values[idx], value_min_max_lst[idx])

            if self.ready_for_next_gumble_phase(simulation_idx):
                # final selection
                for idx in range(batch_size):
                    root, gumble_noise, value_min_max = roots[idx], gumble_noises[idx], value_min_max_lst[idx]
                    self.sequential_halving(root, gumble_noise, value_min_max)
                self.log('change to phase: {}, top m action -> {}'
                             ''.format(self.current_phase, self.current_num_top_actions), verbose=3)

                # obtain the final results and infos
        search_root_values = np.asarray([root.get_value() for root in roots])
        search_root_policies = []
        for root, value_min_max in zip(roots, value_min_max_lst):
            improved_policy = root.get_improved_policy(self.get_transformed_completed_Qs(root, value_min_max))
            search_root_policies.append(improved_policy)
        search_root_policies = np.asarray(search_root_policies)
        search_best_actions = np.asarray([root.selected_children_idx[0] for root in roots])

        if self.verbose:
            self.log('Final Tree:', verbose=1)
            roots[0].print([])
            self.log('search root value -> \t\t {} \n'
                     'search root policy -> \t\t {} \n'
                     'search best action -> \t\t {}'
                     ''.format(search_root_values[0], search_root_policies[0], search_best_actions[0]),
                     verbose=1, iteration_end=True)
        return search_root_values, search_root_policies, search_best_actions, mcts_info

    def search(self, model, batch_size, root_states, root_values, root_policy_logits,
               use_gumble_noise=True, temperature=1.0, verbose=0, **kwargs):
        # preparation
        Node.set_static_attributes(self.discount, self.num_actions)  # set static parameters of MCTS
        roots = [Node(prior=1) for _ in range(batch_size)]          # set root nodes for the batch
        # expand the roots and update the statistics
        for root, state, value, logit in zip(roots, root_states, root_values, root_policy_logits):
            root_reward_hidden = (torch.zeros(1, self.lstm_hidden_size).cuda().float(),
                                  torch.zeros(1, self.lstm_hidden_size).cuda().float())
            if not self.value_prefix:
                root_reward_hidden = None

            root.expand(state, 0, logit, reward_hidden=root_reward_hidden)
            root.estimated_value_lst.append(value)
            root.visit_count += 1
        # save the min and max value of the tree nodes
        value_min_max_lst = [MinMaxStats(self.value_minmax_delta) for _ in range(batch_size)]

        # set gumble noise (during training)
        if use_gumble_noise:
            gumble_noises = np.random.gumbel(0, 1, (batch_size, self.num_actions)) * temperature
        else:
            gumble_noises = np.zeros((batch_size, self.num_actions))

        assert batch_size == len(root_states) == len(root_values)
        self.verbose = verbose
        if self.verbose:
            np.set_printoptions(precision=3)
            assert batch_size == 1

            self.log('Gumble Noise: {}'.format(gumble_noises), verbose=1)

        # search for N iterations
        mcts_info = {}
        for simulation_idx in range(self.num_simulations):
            leaf_nodes = []                                         # leaf node of the tree of the current simulation
            last_actions = []                                       # the chosen action of the leaf node
            current_states = []                                     # the hidden state of the leaf node
            reward_hidden = ([], [])                        # the reward hidden of lstm
            search_paths = []                                       # the nodes along the current search iteration

            self.log('Iteration {} \t'.format(simulation_idx), verbose=2, iteration_begin=True)
            if self.verbose > 1:
                self.log('Tree:', verbose=2)
                roots[0].print([])

            # select action for the roots
            trajectories = []
            for idx in range(batch_size):
                node = roots[idx]                                   # search begins from the root node
                search_path = [node]                                # save the search path from root to leaf
                value_min_max = value_min_max_lst[idx]              # record the min, max value of the tree states

                # search from the root until a leaf unexpanded node
                action = -1
                select_action_lst = []
                while node.is_expanded():
                    action = self.select_action(node, value_min_max, gumble_noises, simulation_idx)
                    node = node.children[action]
                    search_path.append(node)
                    select_action_lst.append(action)

                assert action >= 0

                self.log('selection path -> {}'.format(select_action_lst), verbose=4)
                # update some statistics
                parent = search_path[-2]                            # get the parent of the leaf node
                current_states.append(parent.state)
                reward_hidden[0].append(parent.reward_hidden[0])
                reward_hidden[1].append(parent.reward_hidden[1])

                last_actions.append(action)
                leaf_nodes.append(node)
                search_paths.append(search_path)
                trajectories.append(select_action_lst)

            # inference state, reward, value, policy given the current state
            current_states = torch.stack(current_states, dim=0)
            reward_hidden = (torch.stack(reward_hidden[0], dim=1),
                             torch.stack(reward_hidden[1], dim=1))
            last_actions = torch.from_numpy(np.asarray(last_actions)).cuda().long().unsqueeze(1)
            next_states, next_value_prefixes, next_values, next_logits, reward_hidden = self.update_statistics(
                prediction=True,                                    # use model prediction instead of env simulation
                model=model,                                        # model
                states=current_states,                              # current states
                actions=last_actions,                               # last actions
                reward_hidden=reward_hidden,                        # reward hidden
            )

            # change value prefix to reward
            search_lens = [len(search_path) for search_path in search_paths]
            reset_idx = (np.array(search_lens) % self.lstm_horizon_len == 0)
            if self.value_prefix:
                reward_hidden[0][:, reset_idx, :] = 0
                reward_hidden[1][:, reset_idx, :] = 0
            to_reset_lst = reset_idx.astype(np.int32).tolist()

            # expand the leaf node and backward for statistics update
            for idx in range(batch_size):
                # expand the leaf node
                leaf_nodes[idx].expand(next_states[idx], next_value_prefixes[idx], next_logits[idx],
                                       (reward_hidden[0][0][idx].unsqueeze(0), reward_hidden[1][0][idx].unsqueeze(0)),
                                       to_reset_lst[idx])
                # backward from the leaf node to the root
                self.back_propagate(search_paths[idx], next_values[idx], value_min_max_lst[idx])

            if self.ready_for_next_gumble_phase(simulation_idx):
                # final selection
                for idx in range(batch_size):
                    root, gumble_noise, value_min_max = roots[idx], gumble_noises[idx], value_min_max_lst[idx]
                    self.sequential_halving(root, gumble_noise, value_min_max)
                self.log('change to phase: {}, top m action -> {}'
                             ''.format(self.current_phase, self.current_num_top_actions), verbose=3)

                # obtain the final results and infos
        search_root_values = np.asarray([root.get_value() for root in roots])
        search_root_policies = []
        for root, value_min_max in zip(roots, value_min_max_lst):
            improved_policy = root.get_improved_policy(self.get_transformed_completed_Qs(root, value_min_max))
            search_root_policies.append(improved_policy)
        search_root_policies = np.asarray(search_root_policies)
        search_best_actions = np.asarray([root.selected_children_idx[0] for root in roots])

        if self.verbose:
            self.log('Final Tree:', verbose=1)
            roots[0].print([])
            self.log('search root value -> \t\t {} \n'
                     'search root policy -> \t\t {} \n'
                     'search best action -> \t\t {}'
                     ''.format(search_root_values[0], search_root_policies[0], search_best_actions[0]),
                     verbose=1, iteration_end=True)
        return search_root_values, search_root_policies, search_best_actions, mcts_info

    def sigma_transform(self, max_child_visit_count, value):
        return (self.c_visit + max_child_visit_count) * self.c_scale * value

    def get_transformed_completed_Qs(self, node: Node, value_min_max):
        # get completed Q
        completed_Qs = node.get_completed_Q(value_min_max.normalize)
        # calculate the transformed Q values
        max_child_visit_count = max([child.visit_count for child in node.children])
        transformed_completed_Qs = self.sigma_transform(max_child_visit_count, completed_Qs)
        self.log('Get transformed completed Q...\n'
                 'completed Qs -> \t\t {} \n'
                 'max visit cound of children -> \t {} \n'
                 'transformed completed Qs -> \t {}'
                 ''.format(completed_Qs, max_child_visit_count, transformed_completed_Qs), verbose=4)
        return transformed_completed_Qs


    def select_action(self, node: Node, value_min_max: MinMaxStats, gumbel_noises, simulation_idx):

        def takeSecond(elem):
            return elem[1]

        if node.is_root():
            if simulation_idx == 0:
                children_priors = node.get_children_priors()
                children_scores = []
                for a in range(node.num_actions):
                    children_scores.append((a, gumbel_noises[a] + children_priors[a]))
                children_scores.sort(key=takeSecond, reverse=True)
                for a in range(node.num_actions):
                    node.selected_children_idx.append(children_scores[a][0])

            action = self.do_equal_visit(node)
            self.log('action select at root node, equal visit from {} -> {}'.format(node.selected_children_idx, action),
                     verbose=4)
            return action
        else:
            ## for the non-root nodes, scores are calculated in another way
            # calculate the improved policy
            improved_policy = node.get_improved_policy(self.get_transformed_completed_Qs(node, value_min_max))
            children_visits = node.get_children_visits()
            # calculate the scores for each child
            children_scores = [improved_policy[action] - children_visits[action] / (1 + node.get_children_visit_sum())
                               for action in range(node.num_actions)]
            action = np.argmax(children_scores)
            self.log('action select at non-root node: \n'
                     'improved policy -> \t\t {} \n'
                     'children visits -> \t\t {} \n'
                     'children scores -> \t\t {} \n'
                     'best action -> \t\t\t {} \n'
                     ''.format(improved_policy, children_visits, children_scores, action), verbose=4)
            return action

    def back_propagate(self, search_path, leaf_node_value, value_min_max):
        value = leaf_node_value
        path_len = len(search_path)
        for i in range(path_len - 1, -1, -1):
            node = search_path[i]
            node.estimated_value_lst.append(value)
            node.visit_count += 1

            value = node.get_reward() + self.discount * value
            self.log('Update min max value [{:.3f}, {:.3f}] by {:.3f}'
                     ''.format(value_min_max.minimum, value_min_max.maximum, value), verbose=3)
            value_min_max.update(value)

    def do_equal_visit(self, node: Node):
        min_visit_count = self.num_simulations + 1
        action = -1
        for selected_child_idx in node.selected_children_idx:
            visit_count = node.children[selected_child_idx].visit_count
            if visit_count < min_visit_count:
                action = selected_child_idx
                min_visit_count = visit_count
        assert action >= 0
        return action

    def ready_for_next_gumble_phase(self, simulation_idx):
        ready = (simulation_idx + 1) >= self.visit_num_for_next_phase
        if ready:
            self.current_phase += 1
            self.current_num_top_actions //= 2
            assert self.current_num_top_actions == self.num_top_actions // (2 ** self.current_phase)

            # update the total visit num for the next phase
            n = self.num_simulations
            m = self.num_top_actions
            current_m = self.current_num_top_actions
            # visit n / log2(m) * current_m at current phase
            if current_m > 2:
                extra_visit = np.floor(n / (np.log2(m) * current_m)) * current_m
            else:
                extra_visit = n - self.used_visit_num
            self.used_visit_num += extra_visit
            self.visit_num_for_next_phase += extra_visit
            self.visit_num_for_next_phase = min(self.visit_num_for_next_phase, self.num_simulations)

            self.log('Be ready for the next gumble phase at iteration {}: \n'
                     'current top action num is {}, visit {} times for next phase'
                     ''.format(simulation_idx, current_m, self.visit_num_for_next_phase), verbose=3)
        return ready

    def sequential_halving(self, root, gumble_noise, value_min_max):
        ## update the current selected top m actions for the root
        children_prior = root.get_children_priors()
        if self.current_phase == 0:
            # the first phase: score = g + logits from all children
            children_scores = np.asarray([gumble_noise[action] + children_prior[action]
                                          for action in range(root.num_actions)])
            sorted_action_index = np.argsort(children_scores)[::-1]  # sort the scores from large to small
            # obtain the top m actions
            root.selected_children_idx = sorted_action_index[:self.current_num_top_actions]

            self.log('Do sequential halving at phase {}: \n'
                         'gumble noise -> \t\t {} \n'
                         'child prior -> \t\t\t {} \n'
                         'children scores -> \t\t {} \n'
                         'the selected children indexes -> {}'
                         ''.format(self.current_phase, gumble_noise, children_prior, children_scores,
                                   root.selected_children_idx), verbose=3)
        else:
            assert len(root.selected_children_idx) > 1
            # the later phase: score = g + logits + sigma(hat_q) from the selected children
            # obtain the top m / 2 actions from the m actions
            transformed_completed_Qs = self.get_transformed_completed_Qs(root, value_min_max)
            # selected children index, eg: actions=[4, 1, 2, 5] if action space=8
            selected_children_idx = root.selected_children_idx
            children_scores = np.asarray([gumble_noise[action] + children_prior[action] +
                                          transformed_completed_Qs[action]
                                          for action in selected_children_idx])
            sorted_action_index = np.argsort(children_scores)[::-1]  # sort the scores from large to small
            # eg:   select 2 better action from actions=[4, 1, 2, 5], the sorted_action_index=[2, 0, 1, 3], the
            #       actual action is lst[2, 0] = [2, 4]
            root.selected_children_idx = selected_children_idx[sorted_action_index[:self.current_num_top_actions]]
            self.log('Do sequential halving at phase {}: \n'
                         'selected children -> \t\t {} \n'
                         'gumble noise -> \t\t {} \n'
                         'child prior -> \t\t\t {} \n'
                         'transformed completed Qs -> \t {} \n'
                         'children scores -> \t\t {} \n'
                         'the selected children indexes -> \t\t {}'
                         ''.format(self.current_phase, selected_children_idx, gumble_noise[selected_children_idx],
                                   children_prior[selected_children_idx],
                                   transformed_completed_Qs[selected_children_idx], children_scores,
                                   root.selected_children_idx), verbose=3)

        best_action = root.selected_children_idx[0]
        return best_action
