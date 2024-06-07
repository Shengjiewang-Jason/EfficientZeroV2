# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import time
import ray
import torch
import copy
import gym
import imageio
from PIL import Image, ImageDraw
import numpy as np

from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F

from .base import Worker
from ez import mcts
from ez.envs import make_envs
from ez.utils.distribution import SquashedNormal, TruncatedNormal, ContDist
from ez.utils.format import formalize_obs_lst, DiscreteSupport, LinearSchedule, prepare_obs_lst, allocate_gpu, profile, symexp
from ez.data.trajectory import GameTrajectory
from ez.mcts.cy_mcts import Gumbel_MCTS

@ray.remote(num_gpus=0.03)
# @ray.remote(num_gpus=0.14)
class BatchWorker(Worker):
    def __init__(self, rank, agent, replay_buffer, storage, batch_storage, config):
        super().__init__(rank, agent, replay_buffer, storage, config)

        self.model_update_interval = config.train.reanalyze_update_interval
        self.batch_storage = batch_storage

        self.beta_schedule = LinearSchedule(self.total_steps, initial_p=config.priority.priority_prob_beta, final_p=1.0)
        self.total_transitions = self.config.data.total_transitions
        self.auto_td_steps = self.config.rl.auto_td_steps
        self.td_steps = self.config.rl.td_steps
        self.unroll_steps = self.config.rl.unroll_steps
        self.n_stack = self.config.env.n_stack
        self.discount = self.config.rl.discount
        self.value_support = self.config.model.value_support
        self.action_space_size = self.config.env.action_space_size
        self.batch_size = self.config.train.batch_size
        self.PER_alpha = self.config.priority.priority_prob_alpha
        self.env = self.config.env.env
        self.image_based = self.config.env.image_based
        self.reanalyze_ratio = self.config.train.reanalyze_ratio
        self.value_target = self.config.train.value_target
        self.value_target_type = self.config.model.value_target
        self.GAE_max_steps = self.config.model.GAE_max_steps
        self.episodic = self.config.env.episodic
        self.value_prefix = self.config.model.value_prefix
        self.lstm_horizon_len = self.config.model.lstm_horizon_len
        self.training_steps = self.config.train.training_steps
        self.td_lambda = self.config.rl.td_lambda
        self.gray_scale = self.config.env.gray_scale
        self.obs_shape = self.config.env.obs_shape
        self.trajectory_size = self.config.data.trajectory_size
        self.mixed_value_threshold = self.config.train.mixed_value_threshold
        self.lstm_hidden_size = self.config.model.lstm_hidden_size
        self.cnt = 0

    def concat_trajs(self, items):
        obs_lsts, reward_lsts, policy_lsts, action_lsts, pred_value_lsts, search_value_lsts, \
        bootstrapped_value_lsts = items
        traj_lst = []
        for obs_lst, reward_lst, policy_lst, action_lst, pred_value_lst, search_value_lst, bootstrapped_value_lst in \
                zip(obs_lsts, reward_lsts, policy_lsts, action_lsts, pred_value_lsts, search_value_lsts, bootstrapped_value_lsts):
            # traj = GameTrajectory(**self.config.env, **self.config.rl, **self.config.model)
            traj = GameTrajectory(
                n_stack=self.n_stack, discount=self.discount, gray_scale=self.gray_scale, unroll_steps=self.unroll_steps,
                td_steps=self.td_steps, td_lambda=self.td_lambda, obs_shape=self.obs_shape, max_size=self.trajectory_size,
                image_based=self.image_based, episodic=self.episodic, GAE_max_steps=self.GAE_max_steps
            )
            traj.obs_lst = obs_lst
            traj.reward_lst = reward_lst
            traj.policy_lst = policy_lst
            traj.action_lst = action_lst
            traj.pred_value_lst = pred_value_lst
            traj.search_value_lst = search_value_lst
            traj.bootstrapped_value_lst = bootstrapped_value_lst
            traj_lst.append(traj)
        return traj_lst

    def run(self):
        trained_steps = 0

        # create the model for self-play data collection
        self.model = self.agent.build_model()
        self.latest_model = self.agent.build_model()
        if self.config.eval.analysis_value:
            weights = torch.load(self.config.eval.model_path)
            self.model.load_state_dict(weights)
            print('analysis begin')
        self.model.cuda()
        self.latest_model.cuda()
        if int(torch.__version__[0]) == 2:
            self.model = torch.compile(self.model)
            self.latest_model = torch.compile(self.latest_model)
        self.model.eval()
        self.latest_model.eval()
        self.resume_model()

        # wait for starting to train
        while not ray.get(self.storage.get_start_signal.remote()):
            time.sleep(0.5)

        # begin to make batch
        prev_trained_steps = -10
        while not self.is_finished(trained_steps):
            trained_steps = ray.get(self.storage.get_counter.remote())
            if self.config.ray.single_process:
                if trained_steps <= prev_trained_steps:
                    time.sleep(0.1)
                    continue
                prev_trained_steps = trained_steps
                print(f'reanalyze[{self.rank}] makes batch at step {trained_steps}')
            # get the fresh model weights
            self.get_recent_model(trained_steps, 'reanalyze')
            self.get_latest_model(trained_steps, 'latest')
            # if self.config.model.noisy_net:
            #     self.model.value_policy_model.reset_noise()

            start_time = time.time()
            ray_time = self.make_batch(trained_steps, self.cnt)
            self.cnt += 1
            end_time = time.time()
            # if self.cnt % 100 == 0:
            #     print(f'make batch time={end_time-start_time:.3f}s, ray get time={ray_time:.3f}s')

    # @torch.no_grad()
    # @profile
    def make_batch(self, trained_steps, cnt, real_time=False):
        beta = self.beta_schedule.value(trained_steps)
        batch_size = self.batch_size

        # obtain the batch context from replay buffer
        x = time.time()
        batch_context = ray.get(
            self.replay_buffer.prepare_batch_context.remote(batch_size=batch_size,
                                                            alpha=self.PER_alpha,
                                                            beta=beta,
                                                            rank=self.rank,
                                                            cnt=cnt)
        )
        batch_context, validation_flag = batch_context

        ray_time = time.time() - x
        traj_lst, transition_pos_lst, indices_lst, weights_lst, make_time_lst, transition_num, prior_lst = batch_context

        traj_lst = self.concat_trajs(traj_lst)

        # part of policy will be reanalyzed
        reanalyze_batch_size = batch_size if self.env in ['DMC', 'Gym'] \
            else int(batch_size * self.config.train.reanalyze_ratio)
        assert 0 <= reanalyze_batch_size <= batch_size

        # ==============================================================================================================
        # make inputs
        # ==============================================================================================================
        collected_transitions = ray.get(self.replay_buffer.get_transition_num.remote())
        # make observations, actions and masks (if unrolled steps are out of trajectory)
        obs_lst, action_lst, mask_lst = [], [], []
        top_new_masks = []
        # prepare the inputs of a batch
        for i in range(batch_size):
            traj = traj_lst[i]
            state_index = transition_pos_lst[i]
            sample_idx = indices_lst[i]

            top_new_masks.append(int(sample_idx > collected_transitions - self.mixed_value_threshold))

            if self.env in ['DMC', 'Gym']:
                _actions = traj.action_lst[state_index:state_index + self.unroll_steps]
                _unroll_actions = traj.action_lst[state_index + 1:state_index + 1 + self.unroll_steps]
                # _unroll_actions = traj.action_lst[state_index:state_index + self.unroll_steps]
                _mask = [1. for _ in range(_unroll_actions.shape[0])]
                _mask += [0. for _ in range(self.unroll_steps - len(_mask))]
                _rand_actions = np.zeros((self.unroll_steps - _actions.shape[0], self.action_space_size))
                _actions = np.concatenate((_actions, _rand_actions), axis=0)
            else:
                _actions = traj.action_lst[state_index:state_index + self.unroll_steps].tolist()
                _mask = [1. for _ in range(len(_actions))]
                _mask += [0. for _ in range(self.unroll_steps - len(_mask))]
                _actions += [np.random.randint(0, self.action_space_size) for _ in range(self.unroll_steps - len(_actions))]

            # obtain the input observations
            obs_lst.append(traj.get_index_stacked_obs(state_index, padding=True))
            action_lst.append(_actions)
            mask_lst.append(_mask)

        obs_lst = prepare_obs_lst(obs_lst, self.image_based)
        inputs_batch = [obs_lst, action_lst, mask_lst, indices_lst, weights_lst, make_time_lst, prior_lst]
        for i in range(len(inputs_batch)):
            inputs_batch[i] = np.asarray(inputs_batch[i])

        # ==============================================================================================================
        # make targets
        # ==============================================================================================================

        if self.value_target in ['sarsa', 'mixed', 'max']:
            if self.value_target_type == 'GAE':
                prepare_func = self.prepare_reward_value_gae
            elif self.value_target_type == 'bootstrapped':
                prepare_func = self.prepare_reward_value
            else:
                raise NotImplementedError
        elif self.value_target == 'search':
            prepare_func = self.prepare_reward
        else:
            raise NotImplementedError


        # obtain the value prefix (reward), and the value
        batch_value_prefixes, batch_values, td_steps, pre_calc, value_masks = \
            prepare_func(traj_lst, transition_pos_lst, indices_lst, collected_transitions, trained_steps)

        # obtain the re policy
        if reanalyze_batch_size > 0:
            batch_policies_re, sampled_actions, best_actions, reanalyzed_values, pre_lst, policy_masks = \
                self.prepare_policy_reanalyze(
                trained_steps, traj_lst[:reanalyze_batch_size], transition_pos_lst[:reanalyze_batch_size],
                indices_lst[:reanalyze_batch_size],
                state_lst=pre_calc[0], value_lst=pre_calc[1], policy_lst=pre_calc[2], policy_mask=pre_calc[3]
            )

        else:
            batch_policies_re = []
        # obtain the non-re policy
        if batch_size - reanalyze_batch_size > 0:
            batch_policies_non_re = self.prepare_policy_non_reanalyze(traj_lst[reanalyze_batch_size:],
                                                                      transition_pos_lst[reanalyze_batch_size:])
        else:
            batch_policies_non_re = []
        # concat target policy
        batch_policies = batch_policies_re
        if self.env in ['DMC', 'Gym']:
            batch_best_actions = best_actions.reshape(batch_size, self.unroll_steps + 1,
                                                      self.action_space_size)
        else:
            batch_best_actions = np.asarray(best_actions).reshape(batch_size,
                                                                  self.unroll_steps + 1)

        # target value prefix (reward), value, policy
        if self.env not in ['DMC', 'Gym']:
            batch_actions = np.ones_like(batch_policies)
        else:
            batch_actions = sampled_actions.reshape(
                batch_size, self.unroll_steps + 1, -1, self.action_space_size
            )

        targets_batch = [batch_value_prefixes, batch_values, batch_actions, batch_policies, batch_best_actions, top_new_masks, policy_masks, reanalyzed_values]

        for i in range(len(targets_batch)):
            targets_batch[i] = np.asarray(targets_batch[i])

        # ==============================================================================================================
        # push batch into batch queue
        # ==============================================================================================================
        # full batch data: [obs_lst, other stuffs, target stuffs]
        # batch = [inputs_batch[0], inputs_batch[1:], targets_batch]
        batch = [inputs_batch, targets_batch]

        # log
        self.storage.add_log_scalar.remote({
            'batch_worker/td_step': np.mean(td_steps)
        })

        if real_time:
            return batch
        else:
            # push into batch storage
            self.batch_storage.push(batch)

        return ray_time

    # @profile
    def prepare_reward_value_gae_faster(self, traj_lst, transition_pos_lst, indices_lst, collected_transitions, trained_steps):
        # value prefix (or reward), value
        batch_value_prefixes, batch_values = [], []
        extra = max(0, min(int(1 / (1 - self.td_lambda)), self.GAE_max_steps) - self.unroll_steps - 1)

        # init
        value_obs_lst, td_steps_lst, value_mask, policy_mask = [], [], [], []  # mask: 0 -> out of traj
        # policy_obs_lst, policy_mask = [], []
        zero_obs = traj_lst[0].get_zero_obs(self.n_stack, channel_first=False)

        # get obs_{t+k}
        td_steps = 1
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj)

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            traj_obs = traj.get_index_stacked_obs(state_index, extra=extra + td_steps)
            for current_index in range(state_index, state_index + self.unroll_steps + 1 + extra + td_steps):
                bootstrap_index = current_index

                if not self.episodic:
                    if bootstrap_index <= traj_len:
                        beg_index = bootstrap_index - state_index
                        end_index = beg_index + self.n_stack
                        obs = traj_obs[beg_index:end_index]
                        value_mask.append(1)
                        if bootstrap_index < traj_len:
                            policy_mask.append(1)
                        else:
                            policy_mask.append(0)
                    else:
                        value_mask.append(0)
                        policy_mask.append(0)
                        obs = np.asarray(zero_obs)
                else:
                    if bootstrap_index < traj_len:
                        beg_index = bootstrap_index - state_index
                        end_index = beg_index + self.n_stack
                        obs = traj_obs[beg_index:end_index]
                        value_mask.append(1)
                        policy_mask.append(1)
                    else:
                        value_mask.append(0)
                        policy_mask.append(0)
                        obs = np.asarray(zero_obs)

                value_obs_lst.append(obs)
                td_steps_lst.append(td_steps)

        # reanalyze the bootstrapped value v_{t+k}
        state_lst, value_lst, policy_lst = self.efficient_inference(value_obs_lst, only_value=False)
        # v_{t+k}
        batch_size = len(value_lst)
        value_lst = value_lst.reshape(-1) * (np.array([self.discount for _ in range(batch_size)]) ** td_steps_lst)
        value_lst = value_lst * np.array(value_mask)
        # value_lst = np.zeros_like(value_lst)  # for unit test, remove if training
        td_value_lst = copy.deepcopy(value_lst)
        value_lst = value_lst.tolist()
        td_value_lst = td_value_lst.tolist()

        re_state_lst, re_value_lst, re_policy_lst, re_policy_mask = [], [], [], []
        for i in range(len(state_lst)):
            if i % (self.unroll_steps + extra + 1 + td_steps) < self.unroll_steps + 1:
                re_state_lst.append(state_lst[i].unsqueeze(0))
                re_value_lst.append(value_lst[i])
                re_policy_lst.append(policy_lst[i].unsqueeze(0))
                re_policy_mask.append(policy_mask[i])
        re_state_lst = torch.cat(re_state_lst, dim=0)
        re_value_lst = np.asarray(re_value_lst)
        re_policy_lst = torch.cat(re_policy_lst, dim=0)

        # v_{t} = r + ... + gamma ^ k * v_{t+k}
        value_index = 0
        td_lambdas = []
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj)
            target_values = []
            target_value_prefixs = []

            delta_lambda = 0.1 * (collected_transitions - idx) / self.auto_td_steps
            if self.value_target in ['mixed', 'max']:
                delta_lambda = 0.0
            td_lambda = self.td_lambda - delta_lambda
            td_lambda = np.clip(td_lambda, 0.65, self.td_lambda)
            td_lambdas.append(td_lambda)

            delta = np.zeros(self.unroll_steps + 1 + extra)
            advantage = np.zeros(self.unroll_steps + 1 + extra + 1)
            index = self.unroll_steps + extra
            for current_index in reversed(range(state_index, state_index + self.unroll_steps + 1 + extra)):
                bootstrap_index = current_index + td_steps_lst[value_index + index]
                for i, reward in enumerate(traj.reward_lst[current_index:bootstrap_index]):
                    td_value_lst[value_index + index + td_steps] += reward * self.discount ** i

                delta[index] = td_value_lst[value_index + index + td_steps] - value_lst[value_index + index]
                advantage[index] = delta[index] + self.discount * td_lambda * advantage[index + 1]
                index -= 1

            target_values_tmp = advantage[:self.unroll_steps + 1] + np.asarray(value_lst)[value_index:value_index + self.unroll_steps + 1]

            horizon_id = 0
            value_prefix = 0.0
            for i, current_index in enumerate(range(state_index, state_index + self.unroll_steps + 1)):
                # reset every lstm_horizon_len
                if horizon_id % self.lstm_horizon_len == 0 and self.value_prefix:
                    value_prefix = 0.0
                horizon_id += 1

                if current_index < traj_len:
                    # Since the horizon is small and the discount is close to 1.
                    # Compute the reward sum to approximate the value prefix for simplification
                    if self.value_prefix:
                        value_prefix += traj.reward_lst[current_index]
                    else:
                        value_prefix = traj.reward_lst[current_index]
                    target_value_prefixs.append(value_prefix)
                else:
                    target_value_prefixs.append(value_prefix)
                if self.episodic:
                    if current_index < traj_len:
                        target_values.append(target_values_tmp[i])
                    else:
                        target_values.append(0)
                else:
                    if current_index <= traj_len:
                        target_values.append(target_values_tmp[i])
                    else:
                        target_values.append(0)

            value_index += (self.unroll_steps + 1 + extra + td_steps)
            batch_value_prefixes.append(target_value_prefixs)
            batch_values.append(target_values)

        if self.rank == 0 and self.cnt % 20 == 0:
            print(f'--------------- lambda={np.asarray(td_lambdas).mean():.3f} -------------------')
            self.storage.add_log_scalar.remote({
                'batch_worker/td_lambda': np.asarray(td_lambdas).mean()
            })

        value_index = 0
        value_masks, policy_masks = [], []
        for i, idx in enumerate(indices_lst):
            value_masks.append(int(idx > collected_transitions - self.mixed_value_threshold))
            value_index += (self.unroll_steps + 1 + extra)

        value_masks = np.asarray(value_masks)
        return np.asarray(batch_value_prefixes), np.asarray(batch_values), np.asarray(td_steps_lst).flatten(), \
               (re_state_lst, re_value_lst, re_policy_lst, re_policy_mask), value_masks

    # @profile
    def prepare_reward_value_gae(self, traj_lst, transition_pos_lst, indices_lst, collected_transitions, trained_steps):
        # value prefix (or reward), value
        batch_value_prefixes, batch_values = [], []
        extra = max(0, min(int(1 / (1 - self.td_lambda)), self.GAE_max_steps) - self.unroll_steps - 1)

        # init
        value_obs_lst, td_steps_lst, value_mask = [], [], []  # mask: 0 -> out of traj
        policy_obs_lst, policy_mask = [], []
        zero_obs = traj_lst[0].get_zero_obs(self.n_stack, channel_first=False)

        # get obs_{t+k}
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj)
            td_steps = 1

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            traj_obs = traj.get_index_stacked_obs(state_index + td_steps, extra=extra)
            game_obs = traj.get_index_stacked_obs(state_index, extra=extra)
            for current_index in range(state_index, state_index + self.unroll_steps + 1 + extra):
                bootstrap_index = current_index + td_steps

                if not self.episodic:
                    if bootstrap_index <= traj_len:
                        value_mask.append(1)
                        beg_index = bootstrap_index - (state_index + td_steps)
                        end_index = beg_index + self.n_stack
                        obs = traj_obs[beg_index:end_index]
                    else:
                        value_mask.append(0)
                        obs = np.asarray(zero_obs)
                else:
                    if bootstrap_index < traj_len:
                        value_mask.append(1)
                        beg_index = bootstrap_index - (state_index + td_steps)
                        end_index = beg_index + self.n_stack
                        obs = traj_obs[beg_index:end_index]
                    else:
                        value_mask.append(0)
                        obs = np.asarray(zero_obs)

                value_obs_lst.append(obs)
                td_steps_lst.append(td_steps)

                if current_index < traj_len:
                    policy_mask.append(1)
                    beg_index = current_index - state_index
                    end_index = beg_index + self.n_stack
                    obs = game_obs[beg_index:end_index]
                else:
                    policy_mask.append(0)
                    obs = np.asarray(zero_obs)
                policy_obs_lst.append(obs)

        # reanalyze the bootstrapped value v_{t+k}
        _, value_lst, _ = self.efficient_inference(value_obs_lst, only_value=True)
        state_lst, ori_cur_value_lst, policy_lst = self.efficient_inference(policy_obs_lst, only_value=False)
        # v_{t+k}
        batch_size = len(value_lst)
        value_lst = value_lst.reshape(-1) * (np.array([self.discount for _ in range(batch_size)]) ** td_steps_lst)
        value_lst = value_lst * np.array(value_mask)
        # value_lst = np.zeros_like(value_lst)  # for unit test, remove if training
        value_lst = value_lst.tolist()

        cur_value_lst = ori_cur_value_lst.reshape(-1) * np.array(policy_mask)
        # cur_value_lst = np.zeros_like(cur_value_lst)    # for unit test, remove if training
        cur_value_lst = cur_value_lst.tolist()

        state_lst_cut, ori_cur_value_lst_cut, policy_lst_cut, policy_mask_cut = [], [], [], []
        for i in range(len(state_lst)):
            if i % (self.unroll_steps + extra + 1) < self.unroll_steps + 1:
                state_lst_cut.append(state_lst[i].unsqueeze(0))
                ori_cur_value_lst_cut.append(ori_cur_value_lst[i])
                policy_lst_cut.append(policy_lst[i].unsqueeze(0))
                policy_mask_cut.append(policy_mask[i])
        state_lst_cut = torch.cat(state_lst_cut, dim=0)
        ori_cur_value_lst_cut = np.asarray(ori_cur_value_lst_cut)
        policy_lst_cut = torch.cat(policy_lst_cut, dim=0)

        # v_{t} = r + ... + gamma ^ k * v_{t+k}
        value_index = 0
        td_lambdas = []
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj)
            target_values = []
            target_value_prefixs = []

            delta_lambda = 0.1 * (collected_transitions - idx) / self.auto_td_steps
            if self.value_target in ['mixed', 'max']:
                delta_lambda = 0.0
            td_lambda = self.td_lambda - delta_lambda
            td_lambda = np.clip(td_lambda, 0.65, self.td_lambda)
            td_lambdas.append(td_lambda)

            delta = np.zeros(self.unroll_steps + 1 + extra)
            advantage = np.zeros(self.unroll_steps + 1 + extra + 1)
            index = self.unroll_steps + extra
            for current_index in reversed(range(state_index, state_index + self.unroll_steps + 1 + extra)):
                bootstrap_index = current_index + td_steps_lst[value_index + index]

                for i, reward in enumerate(traj.reward_lst[current_index:bootstrap_index]):
                    value_lst[value_index + index] += reward * self.discount ** i

                delta[index] = value_lst[value_index + index] - cur_value_lst[value_index + index]
                advantage[index] = delta[index] + self.discount * td_lambda * advantage[index + 1]
                index -= 1

            target_values_tmp = advantage[:self.unroll_steps + 1] + np.asarray(cur_value_lst)[value_index:value_index + self.unroll_steps + 1]

            horizon_id = 0
            value_prefix = 0.0
            for i, current_index in enumerate(range(state_index, state_index + self.unroll_steps + 1)):
                # reset every lstm_horizon_len
                if horizon_id % self.lstm_horizon_len == 0 and self.value_prefix:
                    value_prefix = 0.0
                horizon_id += 1

                if current_index < traj_len:
                    # Since the horizon is small and the discount is close to 1.
                    # Compute the reward sum to approximate the value prefix for simplification
                    if self.value_prefix:
                        value_prefix += traj.reward_lst[current_index]
                    else:
                        value_prefix = traj.reward_lst[current_index]
                    target_value_prefixs.append(value_prefix)
                else:
                    target_value_prefixs.append(value_prefix)
                if self.episodic:
                    if current_index < traj_len:
                        target_values.append(target_values_tmp[i])
                    else:
                        target_values.append(0)
                else:
                    if current_index <= traj_len:
                        target_values.append(target_values_tmp[i])
                    else:
                        target_values.append(0)

            value_index += (self.unroll_steps + 1 + extra)
            batch_value_prefixes.append(target_value_prefixs)
            batch_values.append(target_values)

        if self.rank == 0 and self.cnt % 20 == 0:
            print(f'--------------- lambda={np.asarray(td_lambdas).mean():.3f} -------------------')
            self.storage.add_log_scalar.remote({
                'batch_worker/td_lambda': np.asarray(td_lambdas).mean()
            })

        value_index = 0
        value_masks, policy_masks = [], []
        for i, idx in enumerate(indices_lst):
            value_masks.append(int(idx > collected_transitions - self.mixed_value_threshold))
            value_index += (self.unroll_steps + 1 + extra)

        value_masks = np.asarray(value_masks)
        return np.asarray(batch_value_prefixes), np.asarray(batch_values), np.asarray(td_steps_lst).flatten(), \
               (state_lst_cut, ori_cur_value_lst_cut, policy_lst_cut, policy_mask_cut), value_masks


    def prepare_reward(self, traj_lst, transition_pos_lst, indices_lst, collected_transitions, trained_steps):
        # value prefix (or reward), value
        batch_value_prefixes = []

        # v_{t} = r + ... + gamma ^ k * v_{t+k}
        value_index = 0
        top_value_masks = []
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj)
            target_value_prefixs = []

            horizon_id = 0
            value_prefix = 0.0
            top_value_masks.append(int(idx > collected_transitions - self.config.train.start_use_mix_training_steps))
            for current_index in range(state_index, state_index + self.unroll_steps + 1):

                # reset every lstm_horizon_len
                if horizon_id % self.lstm_horizon_len == 0 and self.value_prefix:
                    value_prefix = 0.0
                horizon_id += 1

                if current_index < traj_len:
                    # Since the horizon is small and the discount is close to 1.
                    # Compute the reward sum to approximate the value prefix for simplification
                    if self.value_prefix:
                        value_prefix += traj.reward_lst[current_index]
                    else:
                        value_prefix = traj.reward_lst[current_index]
                    target_value_prefixs.append(value_prefix)
                else:
                    target_value_prefixs.append(value_prefix)

                value_index += 1

            batch_value_prefixes.append(target_value_prefixs)

        value_masks = np.asarray(top_value_masks)
        batch_value_prefixes = np.asarray(batch_value_prefixes)
        batch_values = np.zeros_like(batch_value_prefixes)
        td_steps_lst = np.ones_like(batch_value_prefixes)
        return batch_value_prefixes, np.asarray(batch_values), td_steps_lst.flatten(), \
               (None, None, None, None), value_masks

    def prepare_reward_value(self, traj_lst, transition_pos_lst, indices_lst, collected_transitions, trained_steps):
        # value prefix (or reward), value
        batch_value_prefixes, batch_values = [], []
        # search_values = []

        # init
        value_obs_lst, td_steps_lst, value_mask = [], [], []    # mask: 0 -> out of traj
        zero_obs = traj_lst[0].get_zero_obs(self.n_stack, channel_first=False)

        # get obs_{t+k}
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj)

            # off-policy correction: shorter horizon of td steps
            delta_td = (collected_transitions - idx) // self.auto_td_steps
            if self.value_target in ['mixed', 'max']:
                delta_td = 0
            td_steps = self.td_steps - delta_td
            # td_steps = self.td_steps  # for test off-policy issue
            if not self.episodic:
                td_steps = min(traj_len - state_index, td_steps)
            td_steps = np.clip(td_steps, 1, self.td_steps).astype(np.int32)

            obs_idx = state_index + td_steps

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            traj_obs = traj.get_index_stacked_obs(state_index + td_steps)
            for current_index in range(state_index, state_index + self.unroll_steps + 1):
                if not self.episodic:
                    td_steps = min(traj_len - current_index, td_steps)
                    td_steps = max(td_steps, 1)
                bootstrap_index = current_index + td_steps

                if not self.episodic:
                    if bootstrap_index <= traj_len:
                        value_mask.append(1)
                        beg_index = bootstrap_index - obs_idx
                        end_index = beg_index + self.n_stack
                        obs = traj_obs[beg_index:end_index]
                    else:
                        value_mask.append(0)
                        obs = zero_obs
                else:
                    if bootstrap_index < traj_len:
                        value_mask.append(1)
                        beg_index = bootstrap_index - (state_index + td_steps)
                        end_index = beg_index + self.n_stack
                        obs = traj_obs[beg_index:end_index]
                    else:
                        value_mask.append(0)
                        obs = zero_obs

                value_obs_lst.append(obs)
                td_steps_lst.append(td_steps)

        # reanalyze the bootstrapped value v_{t+k}
        state_lst, value_lst, policy_lst = self.efficient_inference(value_obs_lst, only_value=True)
        batch_size = len(value_lst)
        value_lst = value_lst.reshape(-1) * (np.array([self.discount for _ in range(batch_size)]) ** td_steps_lst)
        value_lst = value_lst * np.array(value_mask)
        # value_lst = np.zeros_like(value_lst)    # for unit test, remove if training
        value_lst = value_lst.tolist()

        # v_{t} = r + ... + gamma ^ k * v_{t+k}
        value_index = 0
        top_value_masks = []
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj)
            target_values = []
            target_value_prefixs = []

            horizon_id = 0
            value_prefix = 0.0
            top_value_masks.append(int(idx > collected_transitions - self.mixed_value_threshold))
            for current_index in range(state_index, state_index + self.unroll_steps + 1):
                bootstrap_index = current_index + td_steps_lst[value_index]

                for i, reward in enumerate(traj.reward_lst[current_index:bootstrap_index]):
                    value_lst[value_index] += reward * self.discount ** i

                # reset every lstm_horizon_len
                if horizon_id % self.lstm_horizon_len == 0 and self.value_prefix:
                    value_prefix = 0.0
                horizon_id += 1

                if current_index < traj_len:
                    # Since the horizon is small and the discount is close to 1.
                    # Compute the reward sum to approximate the value prefix for simplification
                    if self.value_prefix:
                        value_prefix += traj.reward_lst[current_index]
                    else:
                        value_prefix = traj.reward_lst[current_index]
                    target_value_prefixs.append(value_prefix)
                else:
                    target_value_prefixs.append(value_prefix)

                if self.episodic:
                    if current_index < traj_len:
                        target_values.append(value_lst[value_index])
                    else:
                        target_values.append(0)
                else:
                    if current_index <= traj_len:
                        target_values.append(value_lst[value_index])
                    else:
                        target_values.append(0)
                value_index += 1

            batch_value_prefixes.append(target_value_prefixs)
            batch_values.append(target_values)

        value_masks = np.asarray(top_value_masks)
        return np.asarray(batch_value_prefixes), np.asarray(batch_values), np.asarray(td_steps_lst).flatten(), \
               (None, None, None, None), value_masks

    def prepare_policy_non_reanalyze(self, traj_lst, transition_pos_lst):
        # policy
        batch_policies = []

        # load searched policy in self-play
        for traj, state_index in zip(traj_lst, transition_pos_lst):
            traj_len = len(traj)
            target_policies = []

            for current_index in range(state_index, state_index + self.unroll_steps + 1):
                if current_index < traj_len:
                    target_policies.append(traj.policy_lst[current_index])
                else:
                    target_policies.append([0 for _ in range(self.action_space_size)])

            batch_policies.append(target_policies)
        return batch_policies

    def prepare_policy_reanalyze(self, trained_steps, traj_lst, transition_pos_lst, indices_lst, state_lst=None, value_lst=None, policy_lst=None, policy_mask=None):
        # policy
        reanalyzed_values = []
        batch_policies = []

        # init
        if value_lst is None:
            policy_obs_lst, policy_mask = [], []   # mask: 0 -> out of traj
            zero_obs = traj_lst[0].get_zero_obs(self.n_stack, channel_first=False)

            # get obs_{t} instead of obs_{t+k}
            for traj, state_index in zip(traj_lst, transition_pos_lst):
                traj_len = len(traj)

                game_obs = traj.get_index_stacked_obs(state_index)
                for current_index in range(state_index, state_index + self.unroll_steps + 1):

                    if current_index < traj_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + self.n_stack
                        obs = game_obs[beg_index:end_index]
                    else:
                        policy_mask.append(0)
                        obs = np.asarray(zero_obs)
                    policy_obs_lst.append(obs)

            # reanalyze the search policy pi_{t}
            state_lst, value_lst, policy_lst = self.efficient_inference(policy_obs_lst, only_value=False)

        # tree search for policies
        batch_size = len(state_lst)

        # temperature
        temperature = self.agent.get_temperature(trained_steps=trained_steps) #* np.ones((batch_size, 1))
        tree = mcts.names[self.config.mcts.language](
            num_actions=self.action_space_size if self.env == 'Atari' else self.config.mcts.num_sampled_actions,
            discount=self.config.rl.discount,
            env=self.env,
            **self.config.mcts,  # pass mcts related params
            **self.config.model,  # pass the value and reward support params
        )
        if self.env == 'Atari':
            if self.config.mcts.use_gumbel:
                r_values, r_policies, best_actions, _ = tree.search(
                    self.model,
                    # self.latest_model,
                    batch_size, state_lst, value_lst, policy_lst, use_gumble_noise=True, temperature=temperature
                )
            else:
                r_values, r_policies, best_actions, _ = tree.search_ori_mcts(
                    self.model, batch_size, state_lst, value_lst, policy_lst, use_noise=True, temperature=temperature, is_reanalyze=True
                )
            sampled_actions = best_actions
            search_best_indexes = best_actions
        else:

            r_values, r_policies, best_actions, sampled_actions, search_best_indexes, _ = tree.search_continuous(
                    self.model,
                    batch_size, state_lst, value_lst, policy_lst, temperature=temperature,
            ) 

        if self.config.train.optimal_Q:
            r_values = self.efficient_recurrent(state_lst, policy_lst)
            r_values = r_values.reshape(-1) * np.array(policy_mask)
            r_values = r_values.tolist()

        # concat policy
        policy_index = 0
        policy_masks = []
        mismatch_index = []
        for traj, state_index, ind in zip(traj_lst, transition_pos_lst, indices_lst):
            target_policies = []
            search_values = []
            policy_masks.append([])
            for current_index in range(state_index, state_index + self.unroll_steps + 1):
                traj_len = len(traj)

                assert (current_index < traj_len) == (policy_mask[policy_index])
                if policy_mask[policy_index]:
                    target_policies.append(r_policies[policy_index])
                    search_values.append(r_values[policy_index])
                    # mask best-action & pi_prime mismatches
                    if r_policies[policy_index].argmax() != search_best_indexes[policy_index]:
                        policy_mask[policy_index] = 0
                        mismatch_index.append(ind + current_index - state_index)
                else:
                    search_values.append(0.0)
                    if self.env in ['DMC','Gym']:
                        target_policies.append([0 for _ in range(sampled_actions.shape[1])])
                    else:
                        target_policies.append([0 for _ in range(self.action_space_size)])
                policy_masks[-1].append(policy_mask[policy_index])
                policy_index += 1
            batch_policies.append(target_policies)
            reanalyzed_values.append(search_values)

        if self.rank == 0 and self.config.eval.analysis_value:
            new_log_index = trained_steps // 5000
            if new_log_index > self.last_log_index:
                self.last_log_index = new_log_index
                min_idx = np.asarray(indices_lst).argmin()
                r_value = reanalyzed_values[min_idx][0]
                self.storage.add_log_scalar.remote({
                    'batch_worker/search_value': r_value
                })
        policy_masks = np.asarray(policy_masks)
        return batch_policies, sampled_actions, best_actions, reanalyzed_values, (state_lst, value_lst, policy_lst, policy_mask), policy_masks

    @torch.no_grad()
    def imagine_episodes(self, pre_lst, traj_lst, transition_pos_lst, trained_steps, policy='search'):
        length = 1
        times = 3

        # input_obs = np.concatenate([stack_obs for _ in range(times)], axis=0)
        # states, values, policies = self.efficient_inference(input_obs)
        states, values, policies, policy_mask = pre_lst
        states = torch.cat([states for _ in range(times)], dim=0)
        values = np.concatenate([values for _ in range(times)], axis=0)
        policies = torch.cat([policies for _ in range(times)], dim=0)
        reward_hidden = (torch.zeros(1, len(states), self.config.model.lstm_hidden_size).cuda(),
                         torch.zeros(1, len(states), self.config.model.lstm_hidden_size).cuda())
        last_values_prefixes = np.zeros(len(states))
        reward_lst = []
        value_lst = []
        temperature = self.agent.get_temperature(trained_steps=trained_steps) * np.ones((len(states), 1))
        for i in range(length):
            if policy == 'search':
                tree = mcts.names[self.config.mcts.language](
                    num_actions=self.config.env.action_space_size if self.env == 'Atari' else self.config.mcts.num_top_actions,
                    discount=self.config.rl.discount,
                    **self.config.mcts,  # pass mcts related params
                    **self.config.model,  # pass the value and reward support params
                )
                if self.env == 'Atari':
                    if self.config.mcts.use_gumbel:
                        r_values, r_policies, best_actions, _ = tree.search(
                            self.model, len(states), states, values, policies,
                            use_gumble_noise=True, temperature=temperature
                        )
                    else:
                        r_values, r_policies, best_actions, _ = tree.search_ori_mcts(
                            self.model, len(states), states, values, policies, use_noise=True,
                            temperature=temperature, is_reanalyze=True
                        )
                else:
                    r_values, r_policies, best_actions, sampled_actions, _, _ = tree.search_continuous(
                        self.model, len(states), states, values, policies,
                        use_gumble_noise=False, temperature=temperature)


            if policy == 'search':
                actions = torch.from_numpy(np.asarray(best_actions)).cuda().float()
            else:
                if self.env == 'Atari':
                    actions = F.gumbel_softmax(policies, hard=True, dim=-1, tau=1e-4)
                    actions = actions.argmax(dim=-1)
                else:
                    actions = policies[:, :policies.shape[-1]//2]
                actions = actions.unsqueeze(1)

            with autocast():
                states, value_prefixes, values, policies, reward_hidden = \
                    self.model.recurrent_inference(states, actions, reward_hidden)
                values = values.squeeze().detach().cpu().numpy()
                value_lst.append(values)
            if self.value_prefix and (i + 1) % self.lstm_horizon_len == 0:
                reward_hidden = (torch.zeros(1, len(states), self.config.model.lstm_hidden_size).cuda(),
                                 torch.zeros(1, len(states), self.config.model.lstm_hidden_size).cuda())
                true_rewards = value_prefixes.squeeze().detach().cpu().numpy()
                # last_values_prefixes = np.zeros(len(states))
            else:
                true_rewards = value_prefixes.squeeze().detach().cpu().numpy() - last_values_prefixes
                last_values_prefixes = value_prefixes.squeeze().detach().cpu().numpy()

            reward_lst.append(true_rewards)

        value = 0
        for i, reward in enumerate(reward_lst):
            value += reward * (self.config.rl.discount ** i)
        value += (self.config.rl.discount ** length) * value_lst[-1]

        value_reshaped = []
        batch_size = len(states) // times
        for i in range(times):
            value_reshaped.append(value[batch_size * i:batch_size * (i+1)])

        value_reshaped = np.asarray(value_reshaped).mean(0)
        output_values = []
        policy_index = 0
        for traj, state_index in zip(traj_lst, transition_pos_lst):
            imagined_values = []

            for current_index in range(state_index, state_index + self.unroll_steps + 1):
                traj_len = len(traj)

                # assert (current_index < traj_len) == (policy_mask[policy_index])
                if policy_mask[policy_index]:
                    imagined_values.append(value_reshaped[policy_index])
                else:
                    imagined_values.append(0.0)

                policy_index += 1

            output_values.append(imagined_values)

        return np.asarray(output_values)

    def efficient_inference(self, obs_lst, only_value=False, value_idx=0):
        batch_size = len(obs_lst)
        obs_lst = np.asarray(obs_lst)
        state_lst, value_lst, policy_lst = [], [], []
        # split a full batch into slices of mini_infer_size
        mini_batch = self.config.train.mini_batch_size
        slices = np.ceil(batch_size / mini_batch).astype(np.int32)
        with torch.no_grad():
            for i in range(slices):
                beg_index = mini_batch * i
                end_index = mini_batch * (i + 1)
                current_obs = obs_lst[beg_index:end_index]
                current_obs = formalize_obs_lst(current_obs, self.image_based)
                # obtain the statistics at current steps
                with autocast():
                    states, values, policies = self.model.initial_inference(current_obs)

                # process outputs
                values = values.detach().cpu().numpy().flatten()
                # concat
                value_lst.append(values)
                if not only_value:
                    state_lst.append(states)
                    policy_lst.append(policies)

        value_lst = np.concatenate(value_lst)
        if not only_value:
            state_lst = torch.cat(state_lst)
            policy_lst = torch.cat(policy_lst)
        return state_lst, value_lst, policy_lst


# ======================================================================================================================
# batch worker
# ======================================================================================================================
def start_batch_worker(rank, agent, replay_buffer, storage, batch_storage, config):
    """
    Start a GPU batch worker. Call this method remotely.
    """
    worker = BatchWorker.remote(rank, agent, replay_buffer, storage, batch_storage, config)
    print(f"[Batch worker GPU] Starting batch worker GPU {rank} at process {os.getpid()}.")
    worker.run.remote()

def start_batch_worker_cpu(rank, agent, replay_buffer, storage, prebatch_storage, config):
    worker = BatchWorker_CPU.remote(rank, agent, replay_buffer, storage, prebatch_storage, config)
    print(f"[Batch worker CPU] Starting batch worker CPU {rank} at process {os.getpid()}.")
    worker.run.remote()

def start_batch_worker_gpu(rank, agent, replay_buffer, storage, prebatch_storage, batch_storage, config):
    worker = BatchWorker_GPU.remote(rank, agent, replay_buffer, storage, prebatch_storage, batch_storage, config)
    print(f"[Batch worker GPU] Starting batch worker GPU {rank} at process {os.getpid()}.")
    worker.run.remote()