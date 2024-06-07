# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import copy
import os
import time
import ray
import torch
import numpy as np

from torch.nn import L1Loss
from pathlib import Path
from torch.cuda.amp import autocast as autocast

from ez.worker.base import Worker
from ez import mcts
from ez.envs import make_envs, make_env
from ez.utils.format import formalize_obs_lst, DiscreteSupport, allocate_gpu, prepare_obs_lst, symexp
from ez.mcts.cy_mcts import Gumbel_MCTS

# @ray.remote(num_gpus=0.05)
@ray.remote(num_gpus=0.05)
class DataWorker(Worker):
    def __init__(self, rank, agent, replay_buffer, storage, config):
        super().__init__(rank, agent, replay_buffer, storage, config)

        self.model_update_interval = config.train.self_play_update_interval
        self.traj_pool = []
        self.pool_size = 1

        # time.sleep(10000)

    @torch.no_grad()
    def run(self):
        config = self.config

        # create the model for self-play data collection
        self.model = self.agent.build_model()
        self.model.cuda()
        if int(torch.__version__[0]) == 2:
            self.model = torch.compile(self.model)
        self.model.eval()
        self.resume_model()

        # make env
        num_envs = config.data.num_envs
        save_path = Path(config.save_path)
        if config.data.save_video:
            video_path = save_path / 'self_play_videos'
        else:
            video_path = None
        cur_seed = config.env.base_seed

        envs = make_envs(config.env.env, config.env.game, num_envs, cur_seed + self.rank * num_envs,
                         save_path=video_path, episodic_life=config.env.episodic, **config.env)   # prev episodic_life=True

        # initialization
        trained_steps = 0           # current training steps
        collected_transitions = ray.get(self.replay_buffer.get_transition_num.remote())   # total transitions collected
        start_training = False      # is training
        max_transitions = config.data.total_transitions // config.actors.data_worker  # max transitions to collect in this worker
        dones = [False for _ in range(num_envs)]
        traj_len = [0 for _ in range(num_envs)]

        stack_obs_windows, game_trajs = self.agent.init_envs(envs, max_steps=self.config.data.trajectory_size)
        prev_game_trajs = [None for _ in range(num_envs)]  # previous game trajectories (split a full game trajectory into several sub trajectories)

        # log data
        episode_return = [0. for _ in range(num_envs)]

        # while loop for collecting data
        while not self.is_finished(trained_steps):
            trained_steps = ray.get(self.storage.get_counter.remote())
            if not start_training:
                start_training = ray.get(self.storage.get_start_signal.remote())

            # get the fresh model weights
            self.get_recent_model(trained_steps, 'self_play')

            if collected_transitions > max_transitions:
                time.sleep(10)
                continue

            # self-play is faster than training speed or finished
            if start_training and (collected_transitions / max_transitions) > (trained_steps / self.config.train.training_steps):
                time.sleep(1)
                continue

            if self.config.ray.single_process:
                trained_steps = ray.get(self.storage.get_counter.remote())
                if start_training and trained_steps <= prev_train_steps:
                    time.sleep(0.1)
                    continue
                prev_train_steps = trained_steps
                print(f'selfplay[{self.rank}] rollouts at step {trained_steps}, collected transitions {collected_transitions}')

            # print('self-playing')
            # temperature
            temperature = self.agent.get_temperature(trained_steps=trained_steps) #* np.ones((num_envs, 1))

            # stack obs
            current_stacked_obs = formalize_obs_lst(stack_obs_windows, image_based=config.env.image_based)
            # obtain the statistics at current steps
            with autocast():
                states, values, policies = self.model.initial_inference(current_stacked_obs)

            # process outputs
            values = values.detach().cpu().numpy().flatten()

            if collected_transitions % 200 == 0 and self.config.model.noisy_net and self.rank == 0:
                print('*******************************')
                print(f'w_ep={self.model.value_policy_model.pi_net[0].weight_epsilon.mean()}')
                print(f'w_mu={self.model.value_policy_model.pi_net[0].weight_mu.mean()}')
                print(f'w_si={self.model.value_policy_model.pi_net[0].weight_sigma.mean()}')
                print(f'b_ep={self.model.value_policy_model.pi_net[0].bias_epsilon.mean()}')
                print(f'b_mu={self.model.value_policy_model.pi_net[0].bias_mu.mean()}')
                print(f'b_si={self.model.value_policy_model.pi_net[0].bias_sigma.mean()}')

            # tree search for policies
            tree = mcts.names[config.mcts.language](
                # num_actions=config.env.action_space_size if config.env.env == 'Atari' else config.mcts.num_top_actions,
                num_actions=config.env.action_space_size if config.env.env == 'Atari' else config.mcts.num_sampled_actions,
                discount=config.rl.discount,
                env=config.env.env,
                **config.mcts,  # pass mcts related params
                **config.model,  # pass the value and reward support params
            )
            if self.config.env.env == 'Atari':
                if self.config.mcts.use_gumbel:
                    r_values, r_policies, best_actions, _ = tree.search(self.model, num_envs, states, values, policies,
                                                                        # use_gumble_noise=False, # for test search
                                                                        temperature=temperature)
                else:
                    r_values, r_policies, best_actions, _ = tree.search_ori_mcts(self.model, num_envs, states, values, policies,
                                                                                    use_noise=True, temperature=temperature)
            else:
                r_values, r_policies, best_actions, sampled_actions, best_indexes, mcts_info = tree.search_continuous(
                        self.model, num_envs, states, values, policies, temperature=temperature,
                        # use_gumble_noise=True,
                        input_noises=None 
                    )

            # step action in environments
            for i in range(num_envs):
                action = best_actions[i]
                obs, reward, done, info = envs[i].step(action)
                dones[i] = done
                traj_len[i] += 1
                episode_return[i] += info['raw_reward']

                # save data to trajectory buffer
                game_trajs[i].store_search_results(values[i], r_values[i], r_policies[i])
                game_trajs[i].append(action, obs, reward)
                # game_trajs[i].raw_obs_lst.append(obs)
                if self.config.env.env == 'Atari':
                    game_trajs[i].snapshot_lst.append([])
                else:
                    game_trajs[i].snapshot_lst.append([])

                # fresh stack windows
                del stack_obs_windows[i][0]
                stack_obs_windows[i].append(obs)

                # if current trajectory is full; we will save the previous trajectory
                if game_trajs[i].is_full():
                    if prev_game_trajs[i] is not None:
                        self.save_previous_trajectory(i, prev_game_trajs, game_trajs,
                                                      # padding=not dones[i]
                                                      )

                    prev_game_trajs[i] = game_trajs[i]

                    # new trajectory
                    game_trajs[i] = self.agent.new_game(max_steps=self.config.data.trajectory_size)
                    game_trajs[i].init(stack_obs_windows[i])

                    traj_len[i] = 0
    
                # reset an env if done
                if dones[i]:
                    # save the previous trajectory
                    if prev_game_trajs[i] is not None:
                        self.save_previous_trajectory(i, prev_game_trajs, game_trajs,
                                                      # padding=False
                                                      )

                    if len(game_trajs[i]) > 0:
                        # save current trajectory
                        game_trajs[i].pad_over([], [], [], [], [])
                        game_trajs[i].save_to_memory()
                        self.put_trajs(game_trajs[i])

                    # log
                    self.storage.add_log_scalar.remote({
                        'self_play/episode_len': traj_len[i],
                        'self_play/episode_return': episode_return[i],
                        'self_play/temperature': temperature
                    })

                    # reset the finished env and new a env
                    if self.config.env.env == 'DMC':
                        envs[i] = make_env(config.env.env, config.env.game, num_envs, cur_seed + self.rank * num_envs,
                             save_path=video_path, episodic_life=config.env.episodic, **config.env)
                    stacked_obs, traj = self.agent.init_env(envs[i], max_steps=self.config.data.trajectory_size)
                    stack_obs_windows[i] = stacked_obs
                    game_trajs[i] = traj
                    prev_game_trajs[i] = None

                    traj_len[i] = 0
                    episode_return[i] = 0
                collected_transitions += 1


    def save_previous_trajectory(self, idx, prev_game_trajs, game_trajs, padding=True):
        """put the previous game trajectory into the pool if the current trajectory is full
        Parameters
        ----------
        idx: int
            index of the traj to handle
        prev_game_trajs: list
            list of the previous game trajectories
        game_trajs: list
            list of the current game trajectories
        """
        if padding:
            # pad over last block trajectory
            if self.config.model.value_target == 'bootstrapped':
                gap_step = self.config.env.n_stack + self.config.rl.td_steps
            else:
                extra = max(0, min(int(1 / (1 - self.config.rl.td_lambda)), self.config.model.GAE_max_steps) - self.config.rl.unroll_steps - 1)
                gap_step = self.config.env.n_stack + 1 + extra + 1

            beg_index = self.config.env.n_stack
            end_index = beg_index + self.config.rl.unroll_steps

            pad_obs_lst = game_trajs[idx].obs_lst[beg_index:end_index]

            pad_policy_lst = game_trajs[idx].policy_lst[0:self.config.rl.unroll_steps]
            pad_reward_lst = game_trajs[idx].reward_lst[0:gap_step - 1]
            pad_pred_values_lst = game_trajs[idx].pred_value_lst[0:gap_step]
            pad_search_values_lst = game_trajs[idx].search_value_lst[0:gap_step]

            # pad over and save
            prev_game_trajs[idx].pad_over(pad_obs_lst, pad_reward_lst, pad_pred_values_lst, pad_search_values_lst,
                                          pad_policy_lst)
        prev_game_trajs[idx].save_to_memory()
        self.put_trajs(prev_game_trajs[idx])

        # reset last block
        prev_game_trajs[idx] = None

    def put_trajs(self, traj):
        if self.config.priority.use_priority:
            traj_len = len(traj)
            pred_values = torch.from_numpy(np.array(traj.pred_value_lst)).cuda().float()
            # search_values = torch.from_numpy(np.array(traj.search_value_lst)).cuda().float()
            if self.config.model.value_target == 'bootstrapped':
                target_values = torch.from_numpy(np.asarray(traj.get_bootstrapped_value())).cuda().float()
            elif self.config.model.value_target == 'GAE':
                target_values = torch.from_numpy(np.asarray(traj.get_gae_value())).cuda().float()
            else:
                raise NotImplementedError
            priorities = L1Loss(reduction='none')(pred_values[:traj_len], target_values[:traj_len]).detach().cpu().numpy() + self.config.priority.min_prior
        else:
            priorities = None
        self.traj_pool.append(traj)
        # save the game histories and clear the pool
        if len(self.traj_pool) >= self.pool_size:
            self.replay_buffer.save_pools.remote(self.traj_pool, priorities)
            del self.traj_pool[:]

# ======================================================================================================================
# data worker for self-play
# ======================================================================================================================
def start_data_worker(rank, agent, replay_buffer, storage, config):
    """
    Start a data worker. Call this method remotely.
    """
    data_worker = DataWorker.remote(rank, agent, replay_buffer, storage, config)
    data_worker.run.remote()
    print(f'[Data worker] Start data worker {rank} at process {os.getpid()}.')
