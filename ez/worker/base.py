# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import time
import torch
import ray
from ez.data.replay_buffer import ReplayBuffer
from ez.data.global_storage import GlobalStorage


class Worker:
    def __init__(self, rank: int, agent, replay_buffer: ReplayBuffer, storage: GlobalStorage, config):
        self.rank = rank
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.storage = storage
        self.config = config

        self.model = None
        self.latest_model = None
        self.model_update_interval = 0
        self.last_model_index = -1
        self.last_latest_model_index = -1
        self.last_log_index = -1
        self.log_info = {}
        self.total_steps = self.config.train.training_steps + self.config.train.offline_training_steps

    def run(self, **kwargs):
        raise NotImplementedError()

    def get_recent_model(self, trained_steps, model_name):
        assert self.model_update_interval > 0
        assert self.model

        new_model_index = trained_steps // self.model_update_interval
        if new_model_index > self.last_model_index:
            self.last_model_index = new_model_index

            # update model
            weights = ray.get(self.storage.get_weights.remote(model_name))
            self.model.set_weights(weights)
            self.model.cuda()
            self.model.eval()
            if self.config.ray.single_process:
                print('[Update {}] get recent model at step {}'.format(model_name, trained_steps))

    def get_latest_model(self, trained_steps, model_name):
        new_model_index = trained_steps // 30
        if new_model_index > self.last_latest_model_index:
            self.last_latest_model_index = new_model_index
            weights = ray.get(self.storage.get_weights.remote(model_name))
            self.latest_model.set_weights(weights)
            self.latest_model.cuda()
            self.latest_model.eval()

    def resume_model(self):
        load_path = self.config.train.load_model_path
        if os.path.exists(load_path):
            print('[Worker] resume model from path: ', load_path)
            weights = torch.load(load_path)
            self.model.load_state_dict(weights)

    def is_finished(self, trained_steps):
        if trained_steps >= self.total_steps:
            time.sleep(1)
            return True
        else:
            return False

    def reset_log_info(self):
        self.log_info = {}

    def log(self, key, val):
        if not self.log_info.get(key):
            self.log_info['key'] = []
        self.log_info['key'].append(val)
