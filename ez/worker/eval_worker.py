# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import time
import ray
import torch
import logging
import numpy as np

from pathlib import Path
from torch.cuda.amp import autocast as autocast

from .base import Worker
from ez import mcts
from ez.eval import eval

# @ray.remote(num_gpus=0.05)
@ray.remote(num_gpus=0.05)
class EvalWorker(Worker):
    def __init__(self, agent, replay_buffer, storage, config):
        super().__init__(0, agent, replay_buffer, storage, config)

    def run(self):
        model = self.agent.build_model()
        if int(torch.__version__[0]) == 2:
            model = torch.compile(model)
        best_eval_score = float('-inf')
        episodes = 0
        counter = 0
        eval_steps = 27000 #if self.config.env.game in ['CrazyClimber', 'UpNDown', 'DemonAttack', 'Asterix', 'KungFuMaster'] else 3000           # due to time limitation, eval 3000 steps (instead of 27000) during training.
        while not self.is_finished(counter):
            counter = ray.get(self.storage.get_counter.remote())
            if counter >= self.config.train.eval_interval * episodes:
                print('[Eval] Start evaluation at step {}.'.format(counter))

                episodes += 1
                model.set_weights(ray.get(self.storage.get_weights.remote('self_play')))
                model.eval()

                save_path = Path(self.config.save_path) / 'evaluation' / 'step_{}'.format(counter)
                save_path.mkdir(parents=True, exist_ok=True)
                model_path = Path(self.config.save_path) / 'model.p'
                eval_score = eval(self.agent, model, self.config.train.eval_n_episode, save_path, self.config,
                                       max_steps=eval_steps, use_pb=False, verbose=0)
                mean_score = eval_score.mean()
                std_score = eval_score.std()
                min_score = eval_score.min()
                max_score = eval_score.max()

                if mean_score >= best_eval_score:
                    best_eval_score = mean_score
                    self.storage.set_best_score.remote(best_eval_score)
                    torch.save(model.state_dict(), model_path)

                self.storage.set_eval_counter.remote(counter)
                self.storage.add_eval_log_scalar.remote({
                    'eval/mean_score': mean_score,
                    'eval/std_score': std_score,
                    'eval/max_score': max_score,
                    'eval/min_score': min_score
                })

            time.sleep(10)


# ======================================================================================================================
# eval worker
# ======================================================================================================================
def start_eval_worker(agent, replay_buffer, storage, config):
    # start data worker
    eval_worker = EvalWorker.remote(agent, replay_buffer, storage, config)
    eval_worker.run.remote()
