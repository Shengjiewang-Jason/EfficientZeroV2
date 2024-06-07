# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import copy
import os
import time
# import SMOS
import ray
import torch
import wandb
import logging
import random
import numpy as np
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from torch.nn import L1Loss
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from ez.utils.format import get_ddp_model_weights, DiscreteSupport, symexp
from ez.utils.loss import kl_loss, cosine_similarity_loss, continuous_loss, symlog_loss, Value_loss
from ez.data.trajectory import GameTrajectory
from ez.data.augmentation import Transforms

def DDP_setup(**kwargs):
    # set master nod
    os.environ['MASTER_ADDR'] = kwargs.get('address')
    # os.environ['MASTER_PORT'] = kwargs.get('port')

    # initialize the process group
    try:
        dist.init_process_group('nccl', rank=kwargs.get('rank'), world_size=kwargs.get('world_size') * kwargs.get('training_size'))
    except:
        dist.init_process_group('gloo', rank=kwargs.get('rank'), world_size=kwargs.get('world_size') * kwargs.get('training_size'))

    print(f'DDP backend={dist.get_backend()}')

class Agent:
    def __init__(self, config):
        self.config = config
        self.transforms = None
        self.obs_shape = None
        self.input_shape = None
        self.action_space_size = None
        self._update = False
        self.use_ddp = True if config.ddp.world_size > 1 or config.ddp.training_size > 1 else False

    def update_config(self):
        raise NotImplementedError

    def train(self, rank, replay_buffer, storage, batch_storage, logger):
        assert self._update
        # update image augmentation transform
        self.update_augmentation_transform()

        # save path
        model_path = Path(self.config.save_path) / 'models'
        model_path.mkdir(parents=True, exist_ok=True)

        is_main_process = (rank == 0)
        if is_main_process:
            train_logger = logging.getLogger('Train')
            eval_logger = logging.getLogger('Eval')

            train_logger.info('config: {}'.format(self.config))
            train_logger.info('save model in: {}'.format(model_path))

        # prepare model
        model = self.build_model().cuda()
        target_model = self.build_model().cuda()
        # load model
        load_path = self.config.train.load_model_path
        if os.path.exists(load_path):
            if is_main_process:
                train_logger.info('resume model from path: {}'.format(load_path))
            weights = torch.load(load_path)
            storage.set_weights.remote(weights, 'self_play')
            storage.set_weights.remote(weights, 'reanalyze')
            storage.set_weights.remote(weights, 'latest')
            model.load_state_dict(weights)
            target_model.load_state_dict(weights)

        # DDP
        if self.use_ddp:
            model = DDP(model, device_ids=[rank])

        if int(torch.__version__[0]) == 2:
            model = torch.compile(model)
            target_model = torch.compile(target_model)
        model.train()
        target_model.eval()

        # optimizer
        if self.config.optimizer.type == 'SGD':
            optimizer = optim.SGD(model.parameters(),
                                  lr=self.config.optimizer.lr,
                                  weight_decay=self.config.optimizer.weight_decay,
                                  momentum=self.config.optimizer.momentum)
        elif self.config.optimizer.type == 'Adam':
            optimizer = optim.Adam(model.parameters(),
                                   lr=self.config.optimizer.lr,
                                   weight_decay=self.config.optimizer.weight_decay)
        elif self.config.optimizer.type == 'AdamW':
            optimizer = optim.AdamW(model.parameters(),
                                    lr=self.config.optimizer.lr,
                                    weight_decay=self.config.optimizer.weight_decay)
        else:
            raise NotImplementedError

        if self.config.optimizer.lr_decay_type == 'cosine':
            max_steps = self.config.train.training_steps - int(self.config.train.training_steps * self.config.optimizer.lr_warm_up)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps * 3, eta_min=0)
        elif self.config.optimizer.lr_decay_type == 'full_cosine':
            max_steps = self.config.train.training_steps - int(self.config.train.training_steps * self.config.optimizer.lr_warm_up)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps // 2, eta_min=0)
        else:
            scheduler = None

        scaler = GradScaler()

        # wait until collecting enough data to start
        while not (ray.get(replay_buffer.get_transition_num.remote()) >= self.config.train.start_transitions):
            time.sleep(1)
            pass
        print('[Train] Begin training...')

        # set signals for other workers
        if is_main_process:
            storage.set_start_signal.remote()
        step_count = 0

        # Note: the interval of the current model and the target model is between x and 2x. (x = target_model_interval)
        # recent_weights is the param of the target model
        recent_weights = self.get_weights(model)

        # some logs
        total_time = 0
        total_steps = self.config.train.training_steps + self.config.train.offline_training_steps
        if is_main_process:
            pb = tqdm(np.arange(total_steps), leave=True)

        # while loop
        self_play_reteurn = 0.
        traj_num, transition_num = 0, 0
        eval_score, eval_best_score = 0., 0.
        prev_eval_counter = -1
        eval_counter = 0

        while not self.is_finished(step_count):
            start_time = time.time()

            # obtain a batch
            batch = batch_storage.pop()
            end_time1 = time.time()
            if batch is None:
                time.sleep(0.3)
                # print('batch is None')
                continue

            # adjust learning rate
            if is_main_process:
                storage.increase_counter.remote()
            lr = self.adjust_lr(optimizer, step_count, scheduler)

            if is_main_process and step_count % 30 == 0:
                latest_weights = self.get_weights(model)
                storage.set_weights.remote(latest_weights, 'latest')

            # update model for self-play
            if is_main_process and step_count % self.config.train.self_play_update_interval == 0:
                weights = self.get_weights(model)
                storage.set_weights.remote(weights, 'self_play')

            # update model for reanalyzing
            if is_main_process and step_count % self.config.train.reanalyze_update_interval == 0:
                storage.set_weights.remote(recent_weights, 'reanalyze')
                target_model.set_weights(recent_weights)
                target_model.cuda()
                target_model.eval()
                recent_weights = self.get_weights(model)

            if step_count % self.config.train.eval_interval == 0:
                if eval_counter == prev_eval_counter:
                    time.sleep(1)
                    continue

            scalers, log_data = self.update_weights(model, batch, optimizer, replay_buffer, scaler, step_count, target_model=target_model)
            scaler = scalers[0]

            loss_data, other_scalar, other_distribution = log_data

            # TODO: maybe this barrier can be removed
            if self.config.ddp.training_size > 1 or self.config.ddp.world_size > 1:
                dist.barrier()

            # save models
            if is_main_process and step_count % self.config.train.save_ckpt_interval == 0:
                cur_model_path = model_path / 'model_{}.p'.format(step_count)
                torch.save(self.get_weights(model), cur_model_path)

            end_time = time.time()
            total_time += end_time - start_time

            step_count += 1
            avg_time = total_time / step_count
            log_scalars = {}
            log_distribution = {}

            pb_interval = 50
            if is_main_process and step_count % pb_interval == 0:
                left_steps = (self.config.train.training_steps + self.config.train.offline_training_steps - step_count)
                left_time = (left_steps * avg_time) / 3600
                batch_queue_size = batch_storage.get_len()
                train_log_str = '[Train] {}, step {}/{}, {:.3f}h left. lr={:.3f}, avg time={:.3f}s, batchQ={}, '\
                                'self-play return={:.3f}, collect {}/{:.3f}k, eval score={:.3f}/{:.3f}. '\
                                'Loss: reward={:.3f}, value={:.3f}, policy={:.3f}, ' \
                                'consistency={:.3f}, entropy={:.3f}'\
                                ''.format(self.config.env.game, step_count, total_steps, left_time, lr, avg_time,
                                          batch_queue_size, self_play_reteurn, traj_num, transition_num / 1000,
                                          eval_score, eval_best_score, loss_data['loss/value_prefix'],
                                          loss_data['loss/value'], loss_data['loss/policy'],
                                          loss_data['loss/consistency'], loss_data['loss/entropy'])
                # print(f'target policy={batch[-1][-1][0, 0]}')
                pb.set_description(train_log_str)

                pb.update(pb_interval)

                log_scalars.update({
                    'train/step_per_second (s)': end_time - start_time,
                    'train/total time (h)': total_time / 3600,
                    'train/avg time (s)': avg_time,
                    'train/lr': lr,
                    'train/queue size': batch_queue_size
                })

            if is_main_process and step_count % self.config.log.log_interval == 0:
                # train_logger.info(train_log_str)
                # self-play statistics
                eval_scalar, remote_scalar, remote_distribution = ray.get(storage.get_log.remote())
                log_scalars.update(remote_scalar)
                log_distribution.update(remote_distribution)

                if remote_scalar.get('self_play/episode_return'):
                    self_play_reteurn = remote_scalar.get('self_play/episode_return')
                if len(eval_scalar) > 0:
                    eval_score = eval_scalar['eval/mean_score']
                    min_score, max_score = eval_scalar['eval/min_score'], eval_scalar['eval/max_score']
                    eval_counter, eval_best_score = ray.get([storage.get_eval_counter.remote(), storage.get_best_score.remote()])

                    eval_log_str = 'Eval {} at at step {}, score = {:.3f}(min: {:.3f}, max: {:.3f}), ' \
                                   'best score over past evaluation = {:.3f}' \
                                   ''.format(self.config.env.game, eval_counter, eval_score, min_score, max_score,
                                             eval_best_score)
                    eval_logger.info(eval_log_str)
                    # TODO: fix the counter issue
                    # logger.log(eval_scalar, eval_counter)
                    logger.log(eval_scalar, step_count)
                    print('[Eval] ', eval_log_str)

                # replay statistics
                traj_num, transition_num, total_priorities = ray.get([
                    replay_buffer.get_traj_num.remote(), replay_buffer.get_transition_num.remote(), replay_buffer.get_priorities.remote()
                ])
                log_scalars.update({
                    'buffer/total_episode_num': traj_num,
                    'buffer/total_transition_num': transition_num
                })
                log_distribution.update({
                    'dist/priorities_in_buffer': total_priorities,
                })
                log_distribution.update(other_distribution)
                self.log_hist(logger, log_distribution, step_count)

            if step_count % 20000 == 0 and self.config.train.periodic_reset:
                print('-------------------------reset network------------------------------')
                model = self.periodic_reset_model(model)

            # training statistics
            log_scalars.update(loss_data)
            log_scalars.update(other_scalar)
            if step_count > 500 and step_count % 1000 == 0:
                logger.log(log_scalars, step_count)

            traj_num, transition_num, total_priorities = ray.get([
                replay_buffer.get_traj_num.remote(), replay_buffer.get_transition_num.remote(),
                replay_buffer.get_priorities.remote()
            ])
            log_distribution.update({
                'dist/priorities_in_buffer': total_priorities,
            })
            log_distribution.update(other_distribution)


        if is_main_process:
            final_weights = self.get_weights(model)
            storage.set_weights.remote(final_weights, 'self_play')
        else:
            final_weights = None

        return final_weights, model

    def reset_network(self, network):
        for layer in network.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def periodic_reset_model(self, model):
        if self.config.env.image_based:
            # reset prediction_backbone
            self.reset_network(model.value_policy_model.resblocks)

            # # reset policy
            self.reset_network(model.value_policy_model.conv1x1_policy)
            self.reset_network(model.value_policy_model.bn_policy)
            self.reset_network(model.value_policy_model.fc_policy)
            #
            # # reset value
            self.reset_network(model.value_policy_model.conv1x1_values)
            self.reset_network(model.value_policy_model.bn_values)
            self.reset_network(model.value_policy_model.fc_values)

        else:
            self.reset_network(model.value_policy_model.val_resblock)
            self.reset_network(model.value_policy_model.pi_resblock)
            self.reset_network(model.value_policy_model.val_ln)
            self.reset_network(model.value_policy_model.pi_ln)
            self.reset_network(model.value_policy_model.val_net)
            self.reset_network(model.value_policy_model.pi_net)

        return model

    # @profile
    def update_weights(self, model, batch, optimizer, replay_buffer, scaler, step_count, target_model=None):
        target_model.eval()
        # init
        batch_size = self.config.train.batch_size
        image_channel = self.config.env.obs_shape[0] if self.config.env.image_based else self.config.env.obs_shape
        unroll_steps = self.config.rl.unroll_steps
        n_stack = self.config.env.n_stack
        gradient_scale = 1. / unroll_steps
        reward_hidden = self.init_reward_hidden(batch_size)
        loss_data = {}
        other_scalar = {}
        other_distribution = {}

        # obtain the batch data
        inputs_batch, targets_batch = batch
        obs_batch_ori, action_batch, mask_batch, indices, weights_lst, make_time, prior_lst = inputs_batch
        target_value_prefixes, target_values, target_actions, target_policies, target_best_actions, \
            top_value_masks, mismatch_masks, search_values = targets_batch
        target_value_prefixes = target_value_prefixes[:, :unroll_steps]

        # obs_batch_raw: [s_{t - stack} ... s_{t} ... s_{t + unroll}]
        if self.config.env.image_based:
            obs_batch_raw = torch.from_numpy(obs_batch_ori).cuda().float() / 255.
        else:
            obs_batch_raw = torch.from_numpy(obs_batch_ori).cuda().float()

        obs_batch = obs_batch_raw[:, 0: n_stack * image_channel]  # obs_batch: current observation
        obs_target_batch = obs_batch_raw[:, image_channel:]       # obs_target_batch: observation of next steps
        # if self.config.train.use_decorrelation:
        #     obs_batch_all = copy.deepcopy(obs_batch)
        #     for step_i in range(1, unroll_steps + 1):
        #         obs_batch_all = torch.cat((obs_batch_all, obs_batch_raw[:, step_i * image_channel: (step_i + n_stack) * image_channel]), dim=0)

        # augmentation
        obs_batch = self.transform(obs_batch)
        obs_target_batch = self.transform(obs_target_batch)
        # if self.config.train.use_decorrelation:
        #     obs_batch_aug1 = self.transform(obs_batch_all)
        #     obs_batch_aug2 = self.transform(obs_batch_all)

        # others to gpu
        if self.config.env.env in ['DMC', 'Gym']:
            action_batch = torch.from_numpy(action_batch).float().cuda()
        else:
            action_batch = torch.from_numpy(action_batch).cuda().unsqueeze(-1).long()
        mask_batch = torch.from_numpy(mask_batch).cuda().float()
        weights = torch.from_numpy(weights_lst).cuda().float()

        max_value_target = np.array([target_values, search_values]).max(0)

        target_value_prefixes = torch.from_numpy(target_value_prefixes).cuda().float()
        target_values = torch.from_numpy(target_values).cuda().float()
        target_actions = torch.from_numpy(target_actions).cuda().float()
        target_policies = torch.from_numpy(target_policies).cuda().float()
        target_best_actions = torch.from_numpy(target_best_actions).cuda().float()
        top_value_masks = torch.from_numpy(top_value_masks).cuda().float()
        mismatch_masks = torch.from_numpy(mismatch_masks).cuda().float()
        search_values = torch.from_numpy(search_values).cuda().float()
        max_value_target = torch.from_numpy(max_value_target).cuda().float()

        # transform value and reward to support
        target_value_prefixes_support = DiscreteSupport.scalar_to_vector(target_value_prefixes, **self.config.model.reward_support)

        with autocast():
            states, values, policies = model.initial_inference(obs_batch, training=True)

        if self.config.model.value_support.type == 'symlog':
            scaled_value = symexp(values).min(0)[0]
        else:
            scaled_value = DiscreteSupport.vector_to_scalar(values, **self.config.model.value_support).min(0)[0]
        if self.config.env.env in ['DMC', 'Gym']:
            scaled_value = scaled_value.clip(0, 1e5)

        # loss of first step 
        # multi options (Value Loss)
        if self.config.train.value_target == 'sarsa':
            this_target_values = target_values
        elif self.config.train.value_target == 'search':
            this_target_values = search_values
        elif self.config.train.value_target == 'mixed':
            if step_count < self.config.train.start_use_mix_training_steps:
                this_target_values = target_values
            else:
                this_target_values = target_values * top_value_masks.unsqueeze(1).repeat(1, unroll_steps + 1) \
                                     + search_values * (1 - top_value_masks).unsqueeze(1).repeat(1, unroll_steps + 1)
        elif self.config.train.value_target == 'max':
            this_target_values = max_value_target
        else:
            raise NotImplementedError

        # update priority
        fresh_priority = L1Loss(reduction='none')(scaled_value.squeeze(-1), this_target_values[:, 0]).detach().cpu().numpy()
        fresh_priority += self.config.priority.min_prior
        replay_buffer.update_priorities.remote(indices, fresh_priority, make_time)

        value_loss = torch.zeros(batch_size).cuda()
        value_loss += Value_loss(values, this_target_values[:, 0], self.config)
        prev_values = values.clone()

        if self.config.env.env in ['DMC', 'Gym']:
            policy_loss, entropy_loss = continuous_loss(
                policies, target_actions[:, 0], target_policies[:, 0],
                target_best_actions[:, 0],
                distribution_type=self.config.model.policy_distribution
            )
            mu = policies[:, :policies.shape[-1] // 2].detach().cpu().numpy().flatten()
            sigma = policies[:, policies.shape[-1] // 2:].detach().cpu().numpy().flatten()
            other_distribution.update({
                'dist/policy_mu': mu,
                'dist/policy_sigma': sigma,
            })
 
        else:
            policy_loss = kl_loss(policies, target_policies[:, 0])
            entropy_loss = torch.zeros(batch_size).cuda()

        value_prefix_loss = torch.zeros(batch_size).cuda()
        consistency_loss = torch.zeros(batch_size).cuda()
        policy_entropy_loss = torch.zeros(batch_size).cuda()
        policy_entropy_loss -= entropy_loss

        prev_value_prefixes = torch.zeros_like(policy_loss)
        # unroll k steps recurrently
        with autocast():
            for step_i in range(unroll_steps):
                mask = mask_batch[:, step_i]
                states, value_prefixes, values, policies, reward_hidden = model.recurrent_inference(states, action_batch[:, step_i], reward_hidden, training=True)

                beg_index = image_channel * step_i
                end_index = image_channel * (step_i + n_stack)

                # consistency loss
                gt_next_states = model.do_representation(obs_target_batch[:, beg_index:end_index])

                # projection for consistency
                dynamic_states_proj = model.do_projection(states, with_grad=True)
                gt_states_proj = model.do_projection(gt_next_states, with_grad=False)
                consistency_loss += cosine_similarity_loss(dynamic_states_proj, gt_states_proj) * mask
  
                # reward, value, policy loss
                if self.config.model.reward_support.type == 'symlog':
                    value_prefix_loss += symlog_loss(value_prefixes, target_value_prefixes[:, step_i]) * mask
                else:
                    value_prefix_loss += kl_loss(value_prefixes, target_value_prefixes_support[:, step_i]) * mask

                value_loss += Value_loss(values, this_target_values[:, step_i + 1], self.config) * mask

                if self.config.env.env in ['DMC', 'Gym']:
                    policy_loss_i, entropy_loss_i = continuous_loss(
                        policies, target_actions[:, step_i + 1], target_policies[:, step_i + 1],
                        target_best_actions[:, step_i + 1],
                        mask=mask,
                        distribution_type=self.config.model.policy_distribution
                    )
                    policy_loss += policy_loss_i
                    policy_entropy_loss -= entropy_loss_i
                else:
                    policy_loss_i = kl_loss(policies, target_policies[:, step_i + 1]) * mask
                    policy_loss += policy_loss_i

                # set half gradient due to two branches of states
                states.register_hook(lambda grad: grad * 0.5)

                # reset reward hidden
                if self.config.model.value_prefix and (step_i + 1) % self.config.model.lstm_horizon_len == 0:
                    reward_hidden = self.init_reward_hidden(batch_size)

        # total loss
        loss = (value_prefix_loss * self.config.train.reward_loss_coeff
                + value_loss * self.config.train.value_loss_coeff
                + policy_loss * self.config.train.policy_loss_coeff
                + consistency_loss * self.config.train.consistency_coeff)

        if self.config.env.env in ['DMC', 'Gym']:
            loss += policy_entropy_loss * self.config.train.entropy_coeff

        weighted_loss = (weights * loss).mean()

        if weighted_loss.isnan():
            import ipdb
            ipdb.set_trace()
            print('loss nan')

        # backward
        parameters = model.parameters()
        with autocast():
            weighted_loss.register_hook(lambda grad: grad * gradient_scale)
        optimizer.zero_grad()
        scaler.scale(weighted_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, self.config.train.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if self.config.model.noisy_net:
            model.value_policy_model.reset_noise()
            target_model.value_policy_model.reset_noise()

        # log
        loss_data.update({
            'loss/total': loss.mean().item(), 'loss/weighted': weighted_loss.mean().item(),
            'loss/consistency': consistency_loss.mean().item(), 'loss/value_prefix': value_prefix_loss.mean().item(),
            'loss/value': value_loss.mean().item(), 'loss/policy': policy_loss.mean().item(),
            'loss/entropy': policy_entropy_loss.mean().item(),

        })

        other_scalar.update({
            'other_log/target_value_prefix_max': target_value_prefixes.detach().cpu().numpy().max(),
            'other_log/target_value_prefix_min': target_value_prefixes.detach().cpu().numpy().min(),
            'other_log/target_value_prefix_mean': target_value_prefixes.detach().cpu().numpy().mean(),
            'other_log/target_value_mean': target_values.detach().cpu().numpy().mean(),
            'other_log/target_value_max': target_values.detach().cpu().numpy().max(),
            'other_log/target_value_min': target_values.detach().cpu().numpy().min(),
            'other_log/mismatch_num': batch_size * (unroll_steps + 1) - mismatch_masks.sum().detach().cpu().numpy()
        })

        other_distribution.update({
            'dist/recent_priority': fresh_priority,
            'dist/weights': weights.detach().cpu().numpy().flatten(),
            'dist/prior_in_batch': prior_lst,
            'dist/indices': indices.flatten(),
            'dist/mask': mask.detach().cpu().numpy().flatten(),
            'dist/target_policy': target_policies.detach().cpu().numpy().flatten(),
        })

        scalers = [scaler]
        return scalers, (loss_data, other_scalar, other_distribution)

    def get_weights(self, model):
        if self.use_ddp:
            return get_ddp_model_weights(model)
        else:
            return model.get_weights()

    def adjust_lr(self, optimizer, step_count, scheduler):
        lr_warm_step = int(self.config.train.training_steps * self.config.optimizer.lr_warm_up)
        optimize_config = self.config.optimizer

        # adjust learning rate, step lr every lr_decay_steps
        if step_count < lr_warm_step:
            lr = optimize_config.lr * step_count / lr_warm_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if self.config.optimizer.lr_decay_type == 'cosine':
                if scheduler is not None:
                    scheduler.step()
                lr = scheduler.get_last_lr()[0] # return a list
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = optimize_config.lr * optimize_config.lr_decay_rate ** (
                            (step_count - lr_warm_step) // optimize_config.lr_decay_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        return lr

    def log_hist(self, logger, distribution_dict, step_count):
        for key, hist in distribution_dict.items():
            table = wandb.Histogram(hist, num_bins=200)
            logger.log({key: table}, step_count)

    def transform(self, observation):
        if self.transforms is not None:
            return self.transforms(observation)
        else:
            return observation

    def build_model(self):
        raise NotImplementedError

    def update_augmentation_transform(self):
        if self.config.augmentation and self.config.env.image_based:
            self.transforms = Transforms(self.config.augmentation, image_shape=(self.obs_shape[1], self.obs_shape[2]))

    def get_temperature(self, trained_steps):
        if self.config.train.change_temperature:
            total_steps = self.config.train.training_steps + self.config.train.offline_training_steps
            # if self.config.env.env == 'Atari':
            if trained_steps < 0.5 * total_steps:   # prev 0.5
                return 1.0
            elif trained_steps < 0.75 * total_steps:    # prev 0.75
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def init_env(self, env, max_steps):
        assert self._update
        obs = env.reset()

        traj = self.new_game(max_steps)
        stacked_obs = [obs for _ in range(self.config.env.n_stack)]
        traj.init(stacked_obs)

        return stacked_obs, traj

    def init_envs(self, envs, max_steps=None):
        assert self._update

        stacked_obs_lst, game_trajs = [], []
        # initialization for envs, stack [n - 1 zero obs, current obs] for the n-stack obs
        for env in envs:
            stacked_obs, traj = self.init_env(env, max_steps)
            stacked_obs_lst.append(stacked_obs)
            game_trajs.append(traj)
        return stacked_obs_lst, game_trajs

    def init_reward_hidden(self, batch_size):
        if self.config.model.value_prefix:
            reward_hidden = (torch.zeros(1, batch_size, self.config.model.lstm_hidden_size).cuda(),
                             torch.zeros(1, batch_size, self.config.model.lstm_hidden_size).cuda())
        else:
            reward_hidden = None
        return reward_hidden

    def new_game(self, max_steps):
        assert self._update

        traj = GameTrajectory(**self.config.env, **self.config.rl, **self.config.model, trajectory_size=max_steps)
        if max_steps is None:
            traj.set_inf_len()
        return traj

    def is_finished(self, trained_steps):
        if trained_steps >= self.config.train.training_steps + self.config.train.offline_training_steps:
            time.sleep(1)
            return True
        else:
            return False


@ray.remote(num_gpus=0.55)
def train_ddp(agent, rank, replay_buffer, storage, batch_storage, logger):
    print(f'training_rank={rank}')
    if rank == 0:
        wandb_name = agent.config.env.game + '-' + agent.config.wandb.tag
        logger = wandb.init(
            name=wandb_name,
            project=agent.config.wandb.project,
            # config=config,
        )
    assert agent._update
    # update image augmentation transform
    agent.update_augmentation_transform()

    # save path
    model_path = Path(agent.config.save_path) / 'models'
    model_path.mkdir(parents=True, exist_ok=True)

    is_main_process = (rank == 0)
    if is_main_process:
        train_logger = logging.getLogger('Train')
        eval_logger = logging.getLogger('Eval')

        train_logger.info('config: {}'.format(agent.config))
        train_logger.info('save model in: {}'.format(model_path))

    # prepare model
    model = agent.build_model().cuda()
    target_model = agent.build_model().cuda()
    # load model
    load_path = agent.config.train.load_model_path
    if os.path.exists(load_path):
        if is_main_process:
            train_logger.info('resume model from path: {}'.format(load_path))
        weights = torch.load(load_path)
        storage.set_weights.remote(weights, 'self_play')
        storage.set_weights.remote(weights, 'reanalyze')
        storage.set_weights.remote(weights, 'latest')
        model.load_state_dict(weights)
        target_model.load_state_dict(weights)

    # DDP
    if agent.use_ddp:
        DDP_setup(rank=rank, world_size=agent.config.ddp.world_size, training_size=agent.config.ddp.training_size, address='127.0.0.1')
        model = DDP(model, device_ids=[rank])

    if int(torch.__version__[0]) == 2:
        model = torch.compile(model)
        target_model = torch.compile(target_model)
    model.train()
    target_model.eval()

    # optimizer
    if agent.config.optimizer.type == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=agent.config.optimizer.lr,
                              weight_decay=agent.config.optimizer.weight_decay,
                              momentum=agent.config.optimizer.momentum)
    elif agent.config.optimizer.type == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=agent.config.optimizer.lr,
                               weight_decay=agent.config.optimizer.weight_decay)
    elif agent.config.optimizer.type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),
                                lr=agent.config.optimizer.lr,
                                weight_decay=agent.config.optimizer.weight_decay)
    else:
        raise NotImplementedError

    if agent.config.optimizer.lr_decay_type == 'cosine':
        max_steps = agent.config.train.training_steps - int(agent.config.train.training_steps * agent.config.optimizer.lr_warm_up)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps * 3, eta_min=0)
    elif agent.config.optimizer.lr_decay_type == 'full_cosine':
        max_steps = agent.config.train.training_steps - int(agent.config.train.training_steps * agent.config.optimizer.lr_warm_up)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps // 2, eta_min=0)
    else:
        scheduler = None

    scaler = GradScaler()

    # wait until collecting enough data to start
    while not (ray.get(replay_buffer.get_transition_num.remote()) >= agent.config.train.start_transitions):
        time.sleep(1)
        pass
    print('[Train] Begin training...')

    # set signals for other workers
    if is_main_process:
        storage.set_start_signal.remote()
    step_count = 0

    # Note: the interval of the current model and the target model is between x and 2x. (x = target_model_interval)
    # recent_weights is the param of the target model
    recent_weights = agent.get_weights(model)

    # some logs
    total_time = 0
    total_steps = agent.config.train.training_steps + agent.config.train.offline_training_steps
    if is_main_process:
        pb = tqdm(np.arange(total_steps), leave=True)

    # while loop
    self_play_reteurn = 0.
    traj_num, transition_num = 0, 0
    eval_score, eval_best_score = 0., 0.
    while not agent.is_finished(step_count):
        start_time = time.time()

        # obtain a batch
        batch = batch_storage.pop()
        end_time1 = time.time()
        if batch is None:
            time.sleep(0.3)
            # print('batch is None')
            continue

        # adjust learning rate
        if is_main_process:
            storage.increase_counter.remote()
        lr = agent.adjust_lr(optimizer, step_count, scheduler)

        if is_main_process and step_count % 30 == 0:
            latest_weights = agent.get_weights(model)
            ray.get(storage.set_weights.remote(latest_weights, 'latest'))

        # update model for agent-play
        if is_main_process and step_count % agent.config.train.self_play_update_interval == 0:
            weights = agent.get_weights(model)
            storage.set_weights.remote(weights, 'self_play')

        # update model for reanalyzing
        if is_main_process and step_count % agent.config.train.reanalyze_update_interval == 0:
            storage.set_weights.remote(recent_weights, 'reanalyze')
            target_model.set_weights(recent_weights)
            target_model.cuda()
            target_model.eval()
            recent_weights = agent.get_weights(model)


        scalers, log_data = agent.update_weights(model.module, batch, optimizer, replay_buffer, scaler, step_count, target_model=target_model)
        scaler = scalers[0]

        loss_data, other_scalar, other_distribution = log_data

        # TODO: maybe this barrier can be removed
        if agent.config.ddp.training_size > 1 or agent.config.ddp.world_size > 1:
            dist.barrier()

        # save models
        if is_main_process and step_count % agent.config.train.save_ckpt_interval == 0:
            cur_model_path = model_path / 'model_{}.p'.format(step_count)
            torch.save(agent.get_weights(model), cur_model_path)

        end_time = time.time()
        total_time += end_time - start_time

        step_count += 1
        avg_time = total_time / step_count
        log_scalars = {}
        log_distribution = {}

        pb_interval = 50
        if is_main_process and step_count % pb_interval == 0:
            left_steps = (agent.config.train.training_steps + agent.config.train.offline_training_steps - step_count)
            left_time = (left_steps * avg_time) / 3600
            batch_queue_size = batch_storage.get_len()
            train_log_str = '[Train] {}, step {}/{}, {:.3f}h left. lr={:.3f}, avg time={:.3f}s, batchQ={}, '\
                            'agent-play return={:.3f}, collect {}/{:.3f}k, eval score={:.3f}/{:.3f}. '\
                            'Loss: reward={:.3f}, value={:.3f}, policy={:.3f}, ' \
                            'consistency={:.3f}, entropy={:.3f}'\
                            ''.format(agent.config.env.game, step_count, total_steps, left_time, lr, avg_time,
                                      batch_queue_size, self_play_reteurn, traj_num, transition_num / 1000,
                                      eval_score, eval_best_score, loss_data['loss/value_prefix'],
                                      loss_data['loss/value'], loss_data['loss/policy'],
                                      loss_data['loss/consistency'], loss_data['loss/entropy'])
            # print(f'target policy={batch[-1][-1][0, 0]}')
            pb.set_description(train_log_str)

            pb.update(pb_interval)

            log_scalars.update({
                'train/step_per_second (s)': end_time - start_time,
                'train/total time (h)': total_time / 3600,
                'train/avg time (s)': avg_time,
                'train/lr': lr,
                'train/queue size': batch_queue_size
            })

        if is_main_process and step_count % agent.config.log.log_interval == 0:
            # train_logger.info(train_log_str)
            # agent-play statistics
            eval_scalar, remote_scalar, remote_distribution = ray.get(storage.get_log.remote())
            log_scalars.update(remote_scalar)
            log_distribution.update(remote_distribution)

            if remote_scalar.get('self_play/episode_return'):
                self_play_reteurn = remote_scalar.get('self_play/episode_return')
            if len(eval_scalar) > 0:
                # TODO: fix the counter issue
                # logger.log(eval_scalar, eval_counter)
                logger.log(eval_scalar, step_count)

                eval_score = eval_scalar['eval/mean_score']
                min_score, max_score = eval_scalar['eval/min_score'], eval_scalar['eval/max_score']
                eval_counter, eval_best_score = ray.get([storage.get_eval_counter.remote(), storage.get_best_score.remote()])

                eval_log_str = 'Eval {} at at step {}, score = {:.3f}(min: {:.3f}, max: {:.3f}), ' \
                               'best score over past evaluation = {:.3f}' \
                               ''.format(agent.config.env.game, eval_counter, eval_score, min_score, max_score,
                                         eval_best_score)
                eval_logger.info(eval_log_str)
                print('[Eval] ', eval_log_str)

            # replay statistics
            traj_num, transition_num, total_priorities = ray.get([
                replay_buffer.get_traj_num.remote(), replay_buffer.get_transition_num.remote(), replay_buffer.get_priorities.remote()
            ])
            log_scalars.update({
                'buffer/total_episode_num': traj_num,
                'buffer/total_transition_num': transition_num
            })
            log_distribution.update({
                'dist/priorities_in_buffer': total_priorities,
            })
            log_distribution.update(other_distribution)
            agent.log_hist(logger, log_distribution, step_count)

        if step_count % 20000 == 0 and agent.config.train.periodic_reset:
            print('-------------------------reset network------------------------------')
            model = agent.periodic_reset_model(model)

        # training statistics
        log_scalars.update(loss_data)
        log_scalars.update(other_scalar)
        if is_main_process and step_count > 100 and step_count % 1000 == 0:
            logger.log(log_scalars, step_count)

        traj_num, transition_num, total_priorities = ray.get([
            replay_buffer.get_traj_num.remote(), replay_buffer.get_transition_num.remote(),
            replay_buffer.get_priorities.remote()
        ])
        log_distribution.update({
            'dist/priorities_in_buffer': total_priorities,
        })
        log_distribution.update(other_distribution)


    final_weights = agent.get_weights(model)
    storage.set_weights.remote(final_weights, 'self_play')

    return final_weights, model