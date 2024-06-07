# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import time
import numpy as np
import ray
import pickle

@ray.remote
class ReplayBuffer:
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size')
        self.buffer_size = kwargs.get('buffer_size')
        self.top_transitions = kwargs.get('top_transitions')
        self.use_priority = kwargs.get('use_priority')
        self.env = kwargs.get('env')
        self.total_transitions = kwargs.get('total_transitions')

        self.base_idx = 0
        self.clear_time = 0
        self.buffer = []
        self.priorities = []
        self.snapshots = []
        self.transition_idx_look_up = []

    def save_pools(self, traj_pool, priorities):
        # save a list of game histories
        for traj in traj_pool:
            if len(traj) > 0:
                self.save_trajectory(traj, priorities)

    def save_trajectory(self, traj, priorities):
        traj_len = len(traj)
        if priorities is None:
            max_prio = self.priorities.max() if self.buffer else 1
            self.priorities = np.concatenate((self.priorities, [max_prio for _ in range(traj_len)]))
        else:
            assert len(traj) == len(priorities), " priorities should be of same length as the game steps"
            priorities = priorities.copy().reshape(-1)
            max_prio = self.priorities.max() if self.buffer else 1
            self.priorities = np.concatenate((self.priorities, [max(max_prio, priorities.max()) for i in range(traj_len)]))

        for snapshot in traj.snapshot_lst:
            self.snapshots.append(snapshot)

        self.buffer.append(traj)
        self.transition_idx_look_up += [(self.base_idx + len(self.buffer) - 1, step_pos) for step_pos in range(traj_len)]


    def get_item(self, idx):
        traj_idx, state_index = self.transition_idx_look_up[idx]
        traj_idx -= self.base_idx
        traj = self.buffer[traj_idx]

        return traj, state_index

    def prepare_batch_context(self, batch_size, alpha, beta, rank, cnt):

        batch_context = self._prepare_batch_context(batch_size, alpha, beta)
        batch_context = (batch_context, False)

        return batch_context

    def _prepare_batch_context_supervised(self, batch_size, alpha=None, beta=None, is_validation=False, force_uniform=False):
        transition_num = self.get_transition_num()
        if is_validation:
            validation_set = np.arange(int(transition_num * 0.95), transition_num)
            indices_lst = np.random.choice(validation_set, batch_size, replace=False)
            weights_lst = (1 / batch_size) * np.ones_like(indices_lst)
        else:
            # sample data
            if self.use_priority:
                probs = self.priorities ** alpha
            else:
                probs = np.ones_like(self.priorities)
            probs = probs[:int(0.95 * transition_num)]
            probs = probs / probs.sum()

            training_set = np.arange(int(transition_num * 0.95))
            if force_uniform:
                indices_lst = np.random.choice(training_set, batch_size, replace=False)
                weights_lst = (1 / batch_size) * np.ones_like(indices_lst)
            else:
                indices_lst = np.random.choice(training_set, batch_size, p=probs, replace=False)
                weights_lst = (transition_num * probs[indices_lst]) ** (-beta)
                weights_lst = weights_lst / weights_lst.max()

        traj_lst, transition_pos_lst = [], []
        # obtain the
        for idx in indices_lst:
            traj, state_index = self.get_item(idx)
            traj_lst.append(traj)
            transition_pos_lst.append(state_index)

        make_time_lst = [time.time() for _ in range(len(indices_lst))]
        context = [self.split_trajs(traj_lst), transition_pos_lst, indices_lst, weights_lst, make_time_lst,
                   transition_num, self.priorities[indices_lst]]
        return context


    def _prepare_batch_context(self, batch_size, alpha, beta):

        transition_num = self.get_transition_num()
        
        # sample data
        if self.use_priority:
            probs = self.priorities ** alpha
        else:
            probs = np.ones_like(self.priorities)

        # sample the top transitions of the current buffer
        if self.env in ['DMC', 'Gym'] and len(self.priorities) > self.top_transitions:
            idx = int(len(self.priorities) - self.top_transitions)
            probs[:idx] = 0
            self.priorities[:idx] = 0

        probs = probs / probs.sum()

        indices_lst = np.random.choice(transition_num, batch_size, p=probs, replace=False)

        # weight
        weights_lst = (transition_num * probs[indices_lst]) ** (-beta)
        weights_lst = weights_lst / weights_lst.max()
        weights_lst = weights_lst.clip(0.1, 1)    # TODO: try weights clip, prev 0.1

        traj_lst, transition_pos_lst = [], []
        # obtain the
        for idx in indices_lst:
            traj, state_index = self.get_item(idx)
            traj_lst.append(traj)
            transition_pos_lst.append(state_index)

        make_time_lst = [time.time() for _ in range(len(indices_lst))]

        context = [self.split_trajs(traj_lst), transition_pos_lst, indices_lst, weights_lst, make_time_lst, transition_num, self.priorities[indices_lst]]
        return context

    def split_trajs(self, traj_lst):
        obs_lsts, reward_lsts, policy_lsts, action_lsts, pred_value_lsts, search_value_lsts, \
        bootstrapped_value_lsts, snapshot_lsts = [], [], [], [], [], [], [], []
        for traj in traj_lst:
            obs_lsts.append(traj.obs_lst)
            reward_lsts.append(traj.reward_lst)
            policy_lsts.append(traj.policy_lst)
            action_lsts.append(traj.action_lst)
            pred_value_lsts.append(traj.pred_value_lst)
            search_value_lsts.append(traj.search_value_lst)
            bootstrapped_value_lsts.append(traj.bootstrapped_value_lst)
            snapshot_lsts.append(traj.snapshot_lst)
        return [obs_lsts, reward_lsts, policy_lsts, action_lsts, pred_value_lsts, search_value_lsts, bootstrapped_value_lsts,
                # snapshot_lsts
                ]

    def update_root_values(self, batch_indices, search_values, transition_positions, unroll_steps):
        val_idx = 0
        for idx, pos in zip(batch_indices, transition_positions):
            traj_idx, state_index = self.transition_idx_look_up[idx]
            traj_idx -= self.base_idx
            for i in range(unroll_steps + 1):
                self.buffer[traj_idx].search_value_lst.setflags(write=True)
                if pos + i < len(self.buffer[traj_idx].search_value_lst):
                    self.buffer[traj_idx].search_value_lst[pos + i] = search_values[val_idx][i]
            val_idx += 1

    def update_priorities(self, batch_indices, batch_priorities, make_time, mask=None):
        # update the priorities for data still in replay buffer
        if mask is None:
            mask = np.ones(len(batch_indices))
        for i in range(len(batch_indices)):
            # if make_time[i] > self.clear_time:
            assert make_time[i] > self.clear_time
            idx, prio = batch_indices[i], batch_priorities[i]
            if mask[i] == 1:
                self.priorities[idx] = prio

    def get_priorities(self):
        return self.priorities

    def get_snapshots(self, indices_lst):
        selected_snapshots = []
        for idx in indices_lst:
            selected_snapshots.append(self.snapshots[idx])
        return selected_snapshots

    def get_traj_num(self):
        return len(self.buffer)

    def get_transition_num(self):
        assert len(self.transition_idx_look_up) == len(self.priorities)
        assert len(self.priorities) == len(self.snapshots)
        return len(self.transition_idx_look_up)

    def save_buffer(self):
        path = '/workspace/EZ-Codebase/buffer/'
        f_buffer = open(path + 'buffer.b', 'wb')
        pickle.dump(self.buffer, f_buffer)
        f_buffer.close()
        f_priorities = open(path + 'priorities.b', 'wb')
        pickle.dump(self.priorities, f_priorities)
        f_priorities.close()
        f_lookup = open(path + 'lookup.b', 'wb')
        pickle.dump(self.transition_idx_look_up, f_lookup)
        f_lookup.close()
        f_snapshot = open(path + 'snapshots.b', 'wb')
        pickle.dump(self.snapshots, f_snapshot)
        f_snapshot.close()
        return True

    def load_buffer(self):
        path = '/workspace/EZ-Codebase/buffer/'
        f = open(path + 'buffer.b', 'rb')
        self.buffer = pickle.load(f)
        f.close()
        f = open(path + 'priorities.b', 'rb')
        self.priorities = pickle.load(f)
        f.close()
        f = open(path + 'lookup.b', 'rb')
        self.transition_idx_look_up = pickle.load(f)
        f.close()
        f = open(path + 'snapshots.b', 'rb')
        self.snapshots = pickle.load(f)
        f.close()
        return True

# ======================================================================================================================
# replay buffer server
# ======================================================================================================================